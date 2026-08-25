package openai

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
)

const (
	responsesHTTPReplayDefaultTTL = 24 * time.Hour
	// These are storage guardrails, not model context limits. The conversation
	// currently producing a response is never truncated to satisfy them.
	responsesHTTPReplayDefaultMemoryMaxBytes = 128 << 20
	responsesHTTPReplayDefaultDiskMaxBytes   = 2 << 30

	responsesHTTPReplayStateDirEnv = "CLIPROXYAPI_RESPONSES_REPLAY_STATE_DIR"
	responsesHTTPReplayTTLEnv      = "CLIPROXYAPI_RESPONSES_REPLAY_TTL"
	responsesHTTPReplayMaxBytesEnv = "CLIPROXYAPI_RESPONSES_REPLAY_MAX_BYTES"
	responsesHTTPReplayModelsEnv   = "CLIPROXYAPI_RESPONSES_REPLAY_MODELS"

	responsesHTTPReplayNodeVersion = 1
)

type responsesHTTPReplayEntry struct {
	parentResponseID string
	rootResponseID   string
	createdAt        time.Time
	input            []byte
	output           []byte
	storedBytes      int64
}

type responsesHTTPReplayDiskNode struct {
	Version          int             `json:"version"`
	ResponseID       string          `json:"response_id"`
	ParentResponseID string          `json:"parent_response_id,omitempty"`
	CreatedAt        time.Time       `json:"created_at"`
	Input            json.RawMessage `json:"input"`
	Output           json.RawMessage `json:"output"`
}

type responsesHTTPReplayStore struct {
	mu         sync.Mutex
	entries    map[string]responsesHTTPReplayEntry
	stateDir   string
	ttl        time.Duration
	maxBytes   int64
	totalBytes int64
	now        func() time.Time
}

var responsesHTTPReplayCache = newResponsesHTTPReplayStoreFromEnvironment()

func newResponsesHTTPReplayStoreFromEnvironment() *responsesHTTPReplayStore {
	ttl := responsesHTTPReplayDefaultTTL
	if rawTTL := strings.TrimSpace(os.Getenv(responsesHTTPReplayTTLEnv)); rawTTL != "" {
		parsedTTL, err := time.ParseDuration(rawTTL)
		if err != nil || parsedTTL <= 0 {
			log.Warnf("responses HTTP replay: ignoring invalid %s", responsesHTTPReplayTTLEnv)
		} else {
			ttl = parsedTTL
		}
	}

	stateDir := strings.TrimSpace(os.Getenv(responsesHTTPReplayStateDirEnv))
	maxBytes := int64(responsesHTTPReplayDefaultMemoryMaxBytes)
	if stateDir != "" {
		maxBytes = int64(responsesHTTPReplayDefaultDiskMaxBytes)
	}
	maxBytesConfigured := false
	if rawMaxBytes := strings.TrimSpace(os.Getenv(responsesHTTPReplayMaxBytesEnv)); rawMaxBytes != "" {
		parsedMaxBytes, err := strconv.ParseInt(rawMaxBytes, 10, 64)
		if err != nil || parsedMaxBytes < 0 {
			log.Warnf("responses HTTP replay: ignoring invalid %s", responsesHTTPReplayMaxBytesEnv)
		} else {
			maxBytes = parsedMaxBytes
			maxBytesConfigured = true
		}
	}

	store, err := newResponsesHTTPReplayStore(stateDir, ttl, maxBytes)
	if err == nil {
		return store
	}

	// A damaged or unsafe persistence directory must never be guessed through.
	// Keep the API available with a fresh in-memory store; old response IDs then
	// fail closed instead of being reconstructed from untrusted state.
	log.Warnf("responses HTTP replay persistence disabled: %v", err)
	if !maxBytesConfigured {
		maxBytes = int64(responsesHTTPReplayDefaultMemoryMaxBytes)
	}
	store, _ = newResponsesHTTPReplayStore("", ttl, maxBytes)
	return store
}

func newResponsesHTTPReplayStore(stateDir string, ttl time.Duration, maxBytes int64) (*responsesHTTPReplayStore, error) {
	if ttl <= 0 {
		return nil, fmt.Errorf("replay TTL must be positive")
	}
	if maxBytes < 0 {
		return nil, fmt.Errorf("replay max bytes must not be negative")
	}

	store := &responsesHTTPReplayStore{
		entries:  make(map[string]responsesHTTPReplayEntry),
		stateDir: strings.TrimSpace(stateDir),
		ttl:      ttl,
		maxBytes: maxBytes,
		now:      time.Now,
	}
	if store.stateDir == "" {
		return store, nil
	}
	if !filepath.IsAbs(store.stateDir) {
		return nil, fmt.Errorf("replay state directory must be absolute")
	}
	if err := ensurePrivateResponsesHTTPReplayStateDir(store.stateDir); err != nil {
		return nil, err
	}
	if err := store.loadPersistentEntries(); err != nil {
		return nil, err
	}
	return store, nil
}

func ensurePrivateResponsesHTTPReplayStateDir(stateDir string) error {
	info, err := os.Stat(stateDir)
	if os.IsNotExist(err) {
		if errMkdir := os.MkdirAll(stateDir, 0o700); errMkdir != nil {
			return fmt.Errorf("create replay state directory: %w", errMkdir)
		}
		info, err = os.Stat(stateDir)
	}
	if err != nil {
		return fmt.Errorf("inspect replay state directory: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("replay state path is not a directory")
	}
	if info.Mode().Perm()&0o077 != 0 {
		return fmt.Errorf("replay state directory must not be accessible by group or others")
	}
	return nil
}

func responsesHTTPReplayEnabledForRequest(rawJSON []byte) bool {
	var request struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(rawJSON, &request); err != nil {
		return false
	}
	model := strings.TrimSpace(request.Model)
	if model == "" {
		return false
	}
	for _, configured := range strings.Split(os.Getenv(responsesHTTPReplayModelsEnv), ",") {
		configured = strings.TrimSpace(configured)
		if configured == "*" || configured == model {
			return true
		}
	}
	return false
}

func prepareResponsesHTTPReplay(rawJSON []byte) ([]byte, error) {
	if !responsesHTTPReplayEnabledForRequest(rawJSON) {
		return rawJSON, nil
	}
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
	if !responsesHTTPReplayEnabledForRequest(requestJSON) {
		return
	}
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
	if !responsesHTTPReplayEnabledForRequest(requestJSON) {
		return
	}
	responseID = strings.TrimSpace(responseID)
	if responseID == "" {
		return
	}
	requestInput, errInput := responsesHTTPInputArray(requestJSON)
	if errInput != nil {
		return
	}
	previousResponseID, errPrevious := responsesHTTPPreviousResponseID(requestJSON)
	if errPrevious != nil {
		return
	}
	if errPut := responsesHTTPReplayCache.put(responseID, previousResponseID, requestInput, outputJSON); errPut != nil {
		log.Warnf("responses HTTP replay state update failed: %v", errPut)
	}
}

func responsesHTTPPreviousResponseID(rawJSON []byte) (string, error) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(rawJSON, &request); err != nil {
		return "", err
	}
	previousRaw, ok := request["previous_response_id"]
	if !ok {
		return "", nil
	}
	var previousResponseID string
	if err := json.Unmarshal(previousRaw, &previousResponseID); err != nil || strings.TrimSpace(previousResponseID) == "" {
		return "", fmt.Errorf("previous_response_id must be a non-empty string")
	}
	return strings.TrimSpace(previousResponseID), nil
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

func validateResponsesHTTPJSONArray(rawJSON []byte) error {
	var items []json.RawMessage
	if err := json.Unmarshal(rawJSON, &items); err != nil {
		return err
	}
	return nil
}

func (s *responsesHTTPReplayStore) get(responseID string) ([]byte, bool) {
	responseID = strings.TrimSpace(responseID)
	if responseID == "" {
		return nil, false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.cleanupExpiredLocked(s.now()); err != nil {
		log.Warnf("responses HTTP replay cleanup failed: %v", err)
		return nil, false
	}
	if _, ok := s.entries[responseID]; !ok {
		return nil, false
	}

	chain := make([]responsesHTTPReplayDiskNode, 0, 8)
	seen := make(map[string]struct{})
	currentID := responseID
	for currentID != "" {
		if _, duplicate := seen[currentID]; duplicate {
			return nil, false
		}
		seen[currentID] = struct{}{}
		entry, ok := s.entries[currentID]
		if !ok {
			return nil, false
		}
		node, err := s.readNodeLocked(currentID, entry)
		if err != nil {
			log.Warnf("responses HTTP replay state read failed: %v", err)
			return nil, false
		}
		chain = append(chain, node)
		currentID = entry.parentResponseID
	}

	items := make([]json.RawMessage, 0, len(chain)*2)
	for i := len(chain) - 1; i >= 0; i-- {
		var inputItems []json.RawMessage
		if err := json.Unmarshal(chain[i].Input, &inputItems); err != nil {
			return nil, false
		}
		items = append(items, inputItems...)
		var outputItems []json.RawMessage
		if err := json.Unmarshal(chain[i].Output, &outputItems); err != nil {
			return nil, false
		}
		items = append(items, outputItems...)
	}
	merged, err := json.Marshal(items)
	if err != nil {
		return nil, false
	}
	return merged, true
}

func (s *responsesHTTPReplayStore) put(responseID, parentResponseID string, input, output []byte) error {
	responseID = strings.TrimSpace(responseID)
	parentResponseID = strings.TrimSpace(parentResponseID)
	if responseID == "" {
		return fmt.Errorf("response id must not be empty")
	}
	if err := validateResponsesHTTPJSONArray(input); err != nil {
		return fmt.Errorf("invalid replay input: %w", err)
	}
	if err := validateResponsesHTTPJSONArray(output); err != nil {
		return fmt.Errorf("invalid replay output: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.now()
	if err := s.cleanupExpiredLocked(now); err != nil {
		return err
	}
	if _, exists := s.entries[responseID]; exists {
		return fmt.Errorf("duplicate response id")
	}

	rootResponseID := responseID
	if parentResponseID != "" {
		parent, ok := s.entries[parentResponseID]
		if !ok {
			return fmt.Errorf("parent response id is unknown or expired")
		}
		rootResponseID = parent.rootResponseID
	}

	node := responsesHTTPReplayDiskNode{
		Version:          responsesHTTPReplayNodeVersion,
		ResponseID:       responseID,
		ParentResponseID: parentResponseID,
		CreatedAt:        now.UTC(),
		Input:            append(json.RawMessage(nil), input...),
		Output:           append(json.RawMessage(nil), output...),
	}
	storedBytes, errWrite := s.writeNodeLocked(node)
	if errWrite != nil {
		return errWrite
	}

	entry := responsesHTTPReplayEntry{
		parentResponseID: parentResponseID,
		rootResponseID:   rootResponseID,
		createdAt:        node.CreatedAt,
		storedBytes:      storedBytes,
	}
	if s.stateDir == "" {
		entry.input = append([]byte(nil), input...)
		entry.output = append([]byte(nil), output...)
	}
	s.entries[responseID] = entry
	s.totalBytes += storedBytes

	// maxBytes is a storage budget, not a model/context ceiling. Evict older
	// complete conversations first, but never truncate the conversation that
	// just produced this response merely to satisfy the budget.
	if err := s.enforceBudgetLocked(rootResponseID); err != nil {
		return err
	}
	return nil
}

func (s *responsesHTTPReplayStore) writeNodeLocked(node responsesHTTPReplayDiskNode) (int64, error) {
	if s.stateDir == "" {
		return int64(len(node.Input) + len(node.Output)), nil
	}
	data, err := json.Marshal(node)
	if err != nil {
		return 0, fmt.Errorf("marshal replay node: %w", err)
	}
	tmp, err := os.CreateTemp(s.stateDir, ".tmp-replay-*")
	if err != nil {
		return 0, fmt.Errorf("create replay temp file: %w", err)
	}
	tmpName := tmp.Name()
	cleanup := func() {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
	}
	if errChmod := tmp.Chmod(0o600); errChmod != nil {
		cleanup()
		return 0, fmt.Errorf("protect replay temp file: %w", errChmod)
	}
	if _, errWrite := tmp.Write(data); errWrite != nil {
		cleanup()
		return 0, fmt.Errorf("write replay temp file: %w", errWrite)
	}
	if errSync := tmp.Sync(); errSync != nil {
		cleanup()
		return 0, fmt.Errorf("sync replay temp file: %w", errSync)
	}
	if errClose := tmp.Close(); errClose != nil {
		_ = os.Remove(tmpName)
		return 0, fmt.Errorf("close replay temp file: %w", errClose)
	}
	finalPath := filepath.Join(s.stateDir, responsesHTTPReplayNodeFileName(node.ResponseID))
	if errRename := os.Rename(tmpName, finalPath); errRename != nil {
		_ = os.Remove(tmpName)
		return 0, fmt.Errorf("publish replay node: %w", errRename)
	}
	if errSyncDir := syncResponsesHTTPReplayDir(s.stateDir); errSyncDir != nil {
		return 0, errSyncDir
	}
	return int64(len(data)), nil
}

func syncResponsesHTTPReplayDir(stateDir string) error {
	dir, err := os.Open(stateDir)
	if err != nil {
		return fmt.Errorf("open replay state directory: %w", err)
	}
	defer dir.Close()
	if errSync := dir.Sync(); errSync != nil {
		return fmt.Errorf("sync replay state directory: %w", errSync)
	}
	return nil
}

func responsesHTTPReplayNodeFileName(responseID string) string {
	sum := sha256.Sum256([]byte(responseID))
	return hex.EncodeToString(sum[:]) + ".json"
}

func (s *responsesHTTPReplayStore) readNodeLocked(responseID string, entry responsesHTTPReplayEntry) (responsesHTTPReplayDiskNode, error) {
	if s.stateDir == "" {
		return responsesHTTPReplayDiskNode{
			Version:          responsesHTTPReplayNodeVersion,
			ResponseID:       responseID,
			ParentResponseID: entry.parentResponseID,
			CreatedAt:        entry.createdAt,
			Input:            append(json.RawMessage(nil), entry.input...),
			Output:           append(json.RawMessage(nil), entry.output...),
		}, nil
	}

	path := filepath.Join(s.stateDir, responsesHTTPReplayNodeFileName(responseID))
	data, err := os.ReadFile(path)
	if err != nil {
		return responsesHTTPReplayDiskNode{}, fmt.Errorf("read replay node: %w", err)
	}
	var node responsesHTTPReplayDiskNode
	if errUnmarshal := json.Unmarshal(data, &node); errUnmarshal != nil {
		return responsesHTTPReplayDiskNode{}, fmt.Errorf("decode replay node: %w", errUnmarshal)
	}
	if errValidate := validateResponsesHTTPReplayDiskNode(node); errValidate != nil {
		return responsesHTTPReplayDiskNode{}, errValidate
	}
	if node.ResponseID != responseID || node.ParentResponseID != entry.parentResponseID || !node.CreatedAt.Equal(entry.createdAt) {
		return responsesHTTPReplayDiskNode{}, fmt.Errorf("replay node metadata mismatch")
	}
	return node, nil
}

func validateResponsesHTTPReplayDiskNode(node responsesHTTPReplayDiskNode) error {
	if node.Version != responsesHTTPReplayNodeVersion {
		return fmt.Errorf("unsupported replay node version")
	}
	if strings.TrimSpace(node.ResponseID) == "" {
		return fmt.Errorf("replay node response id is empty")
	}
	if node.CreatedAt.IsZero() {
		return fmt.Errorf("replay node timestamp is missing")
	}
	if err := validateResponsesHTTPJSONArray(node.Input); err != nil {
		return fmt.Errorf("replay node input is invalid: %w", err)
	}
	if err := validateResponsesHTTPJSONArray(node.Output); err != nil {
		return fmt.Errorf("replay node output is invalid: %w", err)
	}
	return nil
}

func (s *responsesHTTPReplayStore) loadPersistentEntries() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	dirEntries, err := os.ReadDir(s.stateDir)
	if err != nil {
		return fmt.Errorf("read replay state directory: %w", err)
	}
	for _, dirEntry := range dirEntries {
		if dirEntry.IsDir() {
			continue
		}
		name := dirEntry.Name()
		if strings.HasPrefix(name, ".tmp-replay-") {
			_ = os.Remove(filepath.Join(s.stateDir, name))
			continue
		}
		if !strings.HasSuffix(name, ".json") {
			continue
		}
		path := filepath.Join(s.stateDir, name)
		data, errRead := os.ReadFile(path)
		if errRead != nil {
			return fmt.Errorf("read persisted replay node: %w", errRead)
		}
		var node responsesHTTPReplayDiskNode
		if errDecode := json.Unmarshal(data, &node); errDecode != nil {
			return fmt.Errorf("decode persisted replay node: %w", errDecode)
		}
		if errValidate := validateResponsesHTTPReplayDiskNode(node); errValidate != nil {
			return errValidate
		}
		if name != responsesHTTPReplayNodeFileName(node.ResponseID) {
			return fmt.Errorf("persisted replay node filename mismatch")
		}
		if _, duplicate := s.entries[node.ResponseID]; duplicate {
			return fmt.Errorf("duplicate persisted replay response id")
		}
		s.entries[node.ResponseID] = responsesHTTPReplayEntry{
			parentResponseID: strings.TrimSpace(node.ParentResponseID),
			createdAt:        node.CreatedAt,
			storedBytes:      int64(len(data)),
		}
		s.totalBytes += int64(len(data))
	}

	for responseID := range s.entries {
		rootID, errRoot := s.resolveRootLocked(responseID)
		if errRoot != nil {
			return errRoot
		}
		entry := s.entries[responseID]
		entry.rootResponseID = rootID
		s.entries[responseID] = entry
	}
	if errCleanup := s.cleanupExpiredLocked(s.now()); errCleanup != nil {
		return errCleanup
	}
	return s.enforceBudgetLocked(s.newestRootLocked())
}

func (s *responsesHTTPReplayStore) newestRootLocked() string {
	lastActivity := s.rootLastActivityLocked()
	var newestRoot string
	var newestTime time.Time
	for rootID, lastUsed := range lastActivity {
		if newestRoot == "" || lastUsed.After(newestTime) {
			newestRoot = rootID
			newestTime = lastUsed
		}
	}
	return newestRoot
}

func (s *responsesHTTPReplayStore) resolveRootLocked(responseID string) (string, error) {
	seen := make(map[string]struct{})
	currentID := responseID
	for {
		if _, duplicate := seen[currentID]; duplicate {
			return "", fmt.Errorf("persisted replay parent cycle")
		}
		seen[currentID] = struct{}{}
		entry, ok := s.entries[currentID]
		if !ok {
			return "", fmt.Errorf("persisted replay parent is missing")
		}
		if entry.parentResponseID == "" {
			return currentID, nil
		}
		currentID = entry.parentResponseID
	}
}

func (s *responsesHTTPReplayStore) cleanupExpiredLocked(now time.Time) error {
	if len(s.entries) == 0 {
		return nil
	}
	lastActivity := s.rootLastActivityLocked()
	for rootID, lastUsed := range lastActivity {
		if now.Sub(lastUsed) > s.ttl {
			if err := s.removeConversationLocked(rootID); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *responsesHTTPReplayStore) enforceBudgetLocked(keepRootID string) error {
	if s.maxBytes <= 0 || s.totalBytes <= s.maxBytes {
		return nil
	}
	type rootActivity struct {
		rootID   string
		lastUsed time.Time
	}
	lastActivity := s.rootLastActivityLocked()
	roots := make([]rootActivity, 0, len(lastActivity))
	for rootID, lastUsed := range lastActivity {
		if rootID == keepRootID {
			continue
		}
		roots = append(roots, rootActivity{rootID: rootID, lastUsed: lastUsed})
	}
	sort.Slice(roots, func(i, j int) bool {
		return roots[i].lastUsed.Before(roots[j].lastUsed)
	})
	for _, root := range roots {
		if s.totalBytes <= s.maxBytes {
			break
		}
		if err := s.removeConversationLocked(root.rootID); err != nil {
			return err
		}
	}
	return nil
}

func (s *responsesHTTPReplayStore) rootLastActivityLocked() map[string]time.Time {
	lastActivity := make(map[string]time.Time)
	for _, entry := range s.entries {
		lastUsed, ok := lastActivity[entry.rootResponseID]
		if !ok || entry.createdAt.After(lastUsed) {
			lastActivity[entry.rootResponseID] = entry.createdAt
		}
	}
	return lastActivity
}

func (s *responsesHTTPReplayStore) removeConversationLocked(rootResponseID string) error {
	for responseID, entry := range s.entries {
		if entry.rootResponseID != rootResponseID {
			continue
		}
		if s.stateDir != "" {
			path := filepath.Join(s.stateDir, responsesHTTPReplayNodeFileName(responseID))
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("remove replay node: %w", err)
			}
		}
		s.totalBytes -= entry.storedBytes
		delete(s.entries, responseID)
	}
	if s.totalBytes < 0 {
		s.totalBytes = 0
	}
	if s.stateDir != "" {
		if err := syncResponsesHTTPReplayDir(s.stateDir); err != nil {
			return err
		}
	}
	return nil
}

func resetResponsesHTTPReplayCacheForTest() {
	store, _ := newResponsesHTTPReplayStore("", responsesHTTPReplayDefaultTTL, 0)
	responsesHTTPReplayCache = store
}
