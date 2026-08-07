package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

const (
	responsesHTTPReplayTTL        = 30 * time.Minute
	responsesHTTPReplayMaxEntries = 256
)

type responsesHTTPReplayEntry struct {
	input     []byte
	expiresAt time.Time
}

type responsesHTTPReplayStore struct {
	mu      sync.Mutex
	entries map[string]responsesHTTPReplayEntry
	order   []string
}

var responsesHTTPReplayCache = responsesHTTPReplayStore{
	entries: make(map[string]responsesHTTPReplayEntry),
}

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
		return rawJSON, nil
	}

	previousInput, ok := responsesHTTPReplayCache.get(previousResponseID)
	if !ok {
		return rawJSON, nil
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

func (s *responsesHTTPReplayStore) get(responseID string) ([]byte, bool) {
	responseID = strings.TrimSpace(responseID)
	if responseID == "" {
		return nil, false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.entries[responseID]
	if !ok {
		return nil, false
	}
	if time.Now().After(entry.expiresAt) {
		delete(s.entries, responseID)
		return nil, false
	}
	return append([]byte(nil), entry.input...), true
}

func (s *responsesHTTPReplayStore) put(responseID string, input []byte) {
	responseID = strings.TrimSpace(responseID)
	if responseID == "" || len(input) == 0 {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	for id, entry := range s.entries {
		if now.After(entry.expiresAt) {
			delete(s.entries, id)
		}
	}
	if _, exists := s.entries[responseID]; !exists {
		s.order = append(s.order, responseID)
	}
	for len(s.entries) >= responsesHTTPReplayMaxEntries && len(s.order) > 0 {
		oldest := s.order[0]
		s.order = s.order[1:]
		delete(s.entries, oldest)
	}

	s.entries[responseID] = responsesHTTPReplayEntry{
		input:     append([]byte(nil), input...),
		expiresAt: now.Add(responsesHTTPReplayTTL),
	}
}

func resetResponsesHTTPReplayCacheForTest() {
	responsesHTTPReplayCache.mu.Lock()
	defer responsesHTTPReplayCache.mu.Unlock()
	responsesHTTPReplayCache.entries = make(map[string]responsesHTTPReplayEntry)
	responsesHTTPReplayCache.order = nil
}
