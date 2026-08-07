package openai

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
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
	previousResponseID := strings.TrimSpace(gjson.GetBytes(rawJSON, "previous_response_id").String())
	if previousResponseID == "" {
		return rawJSON, nil
	}

	previousInput, ok := responsesHTTPReplayCache.get(previousResponseID)
	if !ok {
		return rawJSON, nil
	}

	currentInput, errInput := responsesHTTPInputArray(rawJSON)
	if errInput != nil {
		return nil, errInput
	}
	mergedInput, errMerge := mergeResponsesHTTPJSONArrays(previousInput, currentInput)
	if errMerge != nil {
		return nil, errMerge
	}

	updated, errSet := sjson.SetRawBytes(rawJSON, "input", mergedInput)
	if errSet != nil {
		return nil, errSet
	}
	updated, errDelete := sjson.DeleteBytes(updated, "previous_response_id")
	if errDelete != nil {
		return nil, errDelete
	}
	return updated, nil
}

func rememberResponsesHTTPReplay(requestJSON, responseJSON []byte) {
	responseID := strings.TrimSpace(gjson.GetBytes(responseJSON, "id").String())
	output := gjson.GetBytes(responseJSON, "output")
	if responseID == "" || !output.Exists() || !output.IsArray() {
		return
	}
	rememberResponsesHTTPReplayOutput(requestJSON, responseID, []byte(output.Raw))
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
	output := gjson.ParseBytes(outputJSON)
	if !output.IsArray() {
		return
	}
	mergedInput, errMerge := mergeResponsesHTTPJSONArrays(requestInput, outputJSON)
	if errMerge != nil {
		return
	}
	responsesHTTPReplayCache.put(responseID, mergedInput)
}

func responsesHTTPInputArray(rawJSON []byte) ([]byte, error) {
	input := gjson.GetBytes(rawJSON, "input")
	if !input.Exists() {
		return []byte("[]"), nil
	}
	if input.IsArray() {
		return []byte(input.Raw), nil
	}
	if input.Type != gjson.String {
		return nil, fmt.Errorf("responses input must be a string or array")
	}

	item := map[string]any{
		"role": "user",
		"content": []map[string]any{
			{
				"type": "input_text",
				"text": input.String(),
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
	for len(s.entries) >= responsesHTTPReplayMaxEntries {
		if len(s.order) == 0 {
			break
		}
		oldest := s.order[0]
		s.order = s.order[1:]
		if _, exists := s.entries[oldest]; exists {
			delete(s.entries, oldest)
		}
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
