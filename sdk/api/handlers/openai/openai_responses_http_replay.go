package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

const (
	responsesHTTPReplayTTL        = 24 * time.Hour
	responsesHTTPReplayMaxEntries = 256
	responsesHTTPReplayMaxBytes   = 64 << 20
	responsesHTTPReplayFileEnv    = "CLIPROXYAPI_RESPONSES_REPLAY_FILE"
	responsesHTTPReplayFileV1     = 1
)

type responsesHTTPReplayEntry struct {
	input     []byte
	expiresAt time.Time
}

type responsesHTTPReplayStore struct {
	mu      sync.Mutex
	path    string
	entries map[string]responsesHTTPReplayEntry
	order   []string
}

type responsesHTTPReplayPersistedEntry struct {
	Input     json.RawMessage `json:"input"`
	ExpiresAt time.Time       `json:"expires_at"`
}

type responsesHTTPReplayPersistedState struct {
	Version int                                          `json:"version"`
	Order   []string                                     `json:"order"`
	Entries map[string]responsesHTTPReplayPersistedEntry `json:"entries"`
}

func newResponsesHTTPReplayStore(path string) *responsesHTTPReplayStore {
	store := &responsesHTTPReplayStore{
		path:    strings.TrimSpace(path),
		entries: make(map[string]responsesHTTPReplayEntry),
	}
	store.load()
	return store
}

var responsesHTTPReplayCache = newResponsesHTTPReplayStore(os.Getenv(responsesHTTPReplayFileEnv))

func prepareResponsesHTTPReplay(rawJSON []byte) ([]byte, error) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(rawJSON, &request); err != nil {
		return nil, err
	}

	previousRaw, ok := request["previous_response_id"]
	if !ok {
		return rawJSON, nil
	}
	var previousResponseID string
	if err := json.Unmarshal(previousRaw, &previousResponseID); err != nil || strings.TrimSpace(previousResponseID) == "" {
		return nil, fmt.Errorf("previous_response_id must be a non-empty string")
	}

	previousInput, ok := responsesHTTPReplayCache.get(previousResponseID)
	if !ok {
		return nil, fmt.Errorf("unknown or expired previous_response_id")
	}
	currentInput, errInput := responsesHTTPInputArrayRaw(request["input"])
	if errInput != nil {
		return nil, errInput
	}
	mergedInput, errMerge := mergeResponsesHTTPJSONArrays(previousInput, currentInput)
	if errMerge != nil {
		return nil, errMerge
	}

	request["input"] = mergedInput
	delete(request, "previous_response_id")
	return json.Marshal(request)
}

func rememberResponsesHTTPReplay(requestJSON, responseJSON []byte) {
	var response struct {
		ID     string          `json:"id"`
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(responseJSON, &response); err != nil || strings.TrimSpace(response.ID) == "" || len(response.Output) == 0 {
		return
	}
	rememberResponsesHTTPReplayOutput(requestJSON, response.ID, response.Output)
}

func rememberResponsesHTTPReplayOutput(requestJSON []byte, responseID string, outputJSON []byte) {
	responseID = strings.TrimSpace(responseID)
	if responseID == "" {
		return
	}
	requestInput, errInput := responsesHTTPInputArray(requestJSON)
	if errInput != nil {
		return
	}
	mergedInput, errMerge := mergeResponsesHTTPJSONArrays(requestInput, outputJSON)
	if errMerge != nil {
		return
	}
	responsesHTTPReplayCache.put(responseID, mergedInput)
}

func responsesHTTPInputArray(rawJSON []byte) ([]byte, error) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(rawJSON, &request); err != nil {
		return nil, err
	}
	return responsesHTTPInputArrayRaw(request["input"])
}

func responsesHTTPInputArrayRaw(input json.RawMessage) ([]byte, error) {
	input = bytes.TrimSpace(input)
	if len(input) == 0 || bytes.Equal(input, []byte("null")) {
		return []byte("[]"), nil
	}

	var items []json.RawMessage
	if err := json.Unmarshal(input, &items); err == nil {
		return json.Marshal(items)
	}

	var text string
	if err := json.Unmarshal(input, &text); err != nil {
		return nil, fmt.Errorf("responses input must be a string or array")
	}
	item := map[string]any{
		"role": "user",
		"content": []map[string]any{
			{
				"type": "input_text",
				"text": text,
			},
		},
	}
	return json.Marshal([]any{item})
}

func mergeResponsesHTTPJSONArrays(leftJSON, rightJSON []byte) ([]byte, error) {
	var left []json.RawMessage
	if err := json.Unmarshal(leftJSON, &left); err != nil {
		return nil, err
	}
	var right []json.RawMessage
	if err := json.Unmarshal(rightJSON, &right); err != nil {
		return nil, err
	}
	return json.Marshal(append(left, right...))
}

func (s *responsesHTTPReplayStore) load() {
	if s.path == "" {
		return
	}
	data, errRead := os.ReadFile(s.path)
	if errRead != nil {
		if !os.IsNotExist(errRead) {
			logrus.WithError(errRead).Warn("failed to load Responses continuation replay state")
		}
		return
	}

	var state responsesHTTPReplayPersistedState
	if errUnmarshal := json.Unmarshal(data, &state); errUnmarshal != nil {
		logrus.WithError(errUnmarshal).Warn("failed to decode Responses continuation replay state")
		return
	}
	if state.Version != responsesHTTPReplayFileV1 {
		logrus.WithField("version", state.Version).Warn("unsupported Responses continuation replay state version")
		return
	}

	now := time.Now()
	for _, responseID := range state.Order {
		responseID = strings.TrimSpace(responseID)
		persisted, ok := state.Entries[responseID]
		if !ok || responseID == "" || now.After(persisted.ExpiresAt) || len(persisted.Input) == 0 {
			continue
		}
		if _, exists := s.entries[responseID]; exists {
			continue
		}
		var items []json.RawMessage
		if errInput := json.Unmarshal(persisted.Input, &items); errInput != nil {
			continue
		}
		s.entries[responseID] = responsesHTTPReplayEntry{
			input:     append([]byte(nil), persisted.Input...),
			expiresAt: persisted.ExpiresAt,
		}
		s.order = append(s.order, responseID)
	}
	s.pruneLocked(now)
}

func (s *responsesHTTPReplayStore) persistLocked() {
	if s.path == "" {
		return
	}
	if errPersist := s.persistStateLocked(); errPersist != nil {
		logrus.WithError(errPersist).Warn("failed to persist Responses continuation replay state")
	}
}

func (s *responsesHTTPReplayStore) persistStateLocked() error {
	state := responsesHTTPReplayPersistedState{
		Version: responsesHTTPReplayFileV1,
		Order:   append([]string(nil), s.order...),
		Entries: make(map[string]responsesHTTPReplayPersistedEntry, len(s.entries)),
	}
	for responseID, entry := range s.entries {
		state.Entries[responseID] = responsesHTTPReplayPersistedEntry{
			Input:     append(json.RawMessage(nil), entry.input...),
			ExpiresAt: entry.expiresAt,
		}
	}
	data, errMarshal := json.Marshal(state)
	if errMarshal != nil {
		return errMarshal
	}

	dir := filepath.Dir(s.path)
	if errMkdir := os.MkdirAll(dir, 0o700); errMkdir != nil {
		return errMkdir
	}
	tmp, errCreate := os.CreateTemp(dir, ".responses-http-replay-*")
	if errCreate != nil {
		return errCreate
	}
	tmpName := tmp.Name()
	keep := false
	defer func() {
		_ = tmp.Close()
		if !keep {
			_ = os.Remove(tmpName)
		}
	}()
	if errChmod := tmp.Chmod(0o600); errChmod != nil {
		return errChmod
	}
	if _, errWrite := tmp.Write(data); errWrite != nil {
		return errWrite
	}
	if errSync := tmp.Sync(); errSync != nil {
		return errSync
	}
	if errClose := tmp.Close(); errClose != nil {
		return errClose
	}
	if errRename := os.Rename(tmpName, s.path); errRename != nil {
		return errRename
	}
	keep = true
	return nil
}

func (s *responsesHTTPReplayStore) pruneLocked(now time.Time) {
	compacted := make([]string, 0, len(s.order))
	seen := make(map[string]struct{}, len(s.entries))
	totalBytes := 0
	for _, responseID := range s.order {
		entry, ok := s.entries[responseID]
		if !ok || now.After(entry.expiresAt) {
			delete(s.entries, responseID)
			continue
		}
		if _, duplicate := seen[responseID]; duplicate {
			continue
		}
		seen[responseID] = struct{}{}
		compacted = append(compacted, responseID)
		totalBytes += len(entry.input)
	}
	s.order = compacted

	for len(s.order) > responsesHTTPReplayMaxEntries || totalBytes > responsesHTTPReplayMaxBytes {
		oldest := s.order[0]
		s.order = s.order[1:]
		if entry, ok := s.entries[oldest]; ok {
			totalBytes -= len(entry.input)
			delete(s.entries, oldest)
		}
	}
}

func (s *responsesHTTPReplayStore) get(responseID string) ([]byte, bool) {
	responseID = strings.TrimSpace(responseID)
	if responseID == "" {
		return nil, false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	entry, ok := s.entries[responseID]
	if !ok {
		return nil, false
	}
	if now.After(entry.expiresAt) {
		delete(s.entries, responseID)
		s.pruneLocked(now)
		s.persistLocked()
		return nil, false
	}
	return append([]byte(nil), entry.input...), true
}

func (s *responsesHTTPReplayStore) put(responseID string, input []byte) {
	responseID = strings.TrimSpace(responseID)
	if responseID == "" || len(input) == 0 || len(input) > responsesHTTPReplayMaxBytes {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	if _, exists := s.entries[responseID]; !exists {
		s.order = append(s.order, responseID)
	}
	s.entries[responseID] = responsesHTTPReplayEntry{
		input:     append([]byte(nil), input...),
		expiresAt: now.Add(responsesHTTPReplayTTL),
	}
	s.pruneLocked(now)
	s.persistLocked()
}

func resetResponsesHTTPReplayCacheForTest() {
	responsesHTTPReplayCache.mu.Lock()
	defer responsesHTTPReplayCache.mu.Unlock()
	responsesHTTPReplayCache.path = ""
	responsesHTTPReplayCache.entries = make(map[string]responsesHTTPReplayEntry)
	responsesHTTPReplayCache.order = nil
}
