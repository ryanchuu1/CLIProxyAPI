package openai

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tidwall/gjson"
)

func TestResponsesHTTPReplayContinuation(t *testing.T) {
	resetResponsesHTTPReplayCacheForTest()
	t.Cleanup(resetResponsesHTTPReplayCacheForTest)

	firstRequest := []byte(`{"model":"gpt-test","input":"Remember blue","stream":false}`)
	firstResponse := []byte(`{"id":"resp_1","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"blue"}]}]}`)
	rememberResponsesHTTPReplay(firstRequest, firstResponse)

	secondRequest := []byte(`{"model":"gpt-test","previous_response_id":"resp_1","input":"What color?","stream":false}`)
	replayed, errReplay := prepareResponsesHTTPReplay(secondRequest)
	if errReplay != nil {
		t.Fatalf("prepare replay: %v", errReplay)
	}
	if gjson.GetBytes(replayed, "previous_response_id").Exists() {
		t.Fatalf("previous_response_id must be removed after local replay: %s", replayed)
	}
	input := gjson.GetBytes(replayed, "input")
	if !input.IsArray() || len(input.Array()) != 3 {
		t.Fatalf("expected prior user + assistant + current user input, got: %s", replayed)
	}
	if got := input.Array()[0].Get("content.0.text").String(); got != "Remember blue" {
		t.Fatalf("unexpected first user input: %q", got)
	}
	if got := input.Array()[1].Get("content.0.text").String(); got != "blue" {
		t.Fatalf("unexpected assistant output: %q", got)
	}
	if got := input.Array()[2].Get("content.0.text").String(); got != "What color?" {
		t.Fatalf("unexpected current user input: %q", got)
	}

	secondResponse := []byte(`{"id":"resp_2","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Blue"}]}]}`)
	rememberResponsesHTTPReplay(replayed, secondResponse)
	thirdRequest := []byte(`{"model":"gpt-test","previous_response_id":"resp_2","input":"Repeat it","stream":false}`)
	thirdReplay, errThird := prepareResponsesHTTPReplay(thirdRequest)
	if errThird != nil {
		t.Fatalf("prepare third-turn replay: %v", errThird)
	}
	if got := len(gjson.GetBytes(thirdReplay, "input").Array()); got != 5 {
		t.Fatalf("expected five transcript items on third turn, got %d: %s", got, thirdReplay)
	}
}

func TestResponsesHTTPReplayUnknownIDFailsClosed(t *testing.T) {
	resetResponsesHTTPReplayCacheForTest()
	t.Cleanup(resetResponsesHTTPReplayCacheForTest)

	raw := []byte(`{"model":"gpt-test","previous_response_id":"resp_missing","input":"hello"}`)
	if _, errPrepare := prepareResponsesHTTPReplay(raw); errPrepare == nil || !strings.Contains(errPrepare.Error(), "unknown or expired previous_response_id") {
		t.Fatalf("unknown response id must fail closed, got: %v", errPrepare)
	}
}

func TestResponsesHTTPReplayWithoutPreviousIDIsUnchanged(t *testing.T) {
	raw := []byte(`{"model":"gpt-test","input":[{"role":"user","content":"hello"}],"stream":false}`)
	got, errPrepare := prepareResponsesHTTPReplay(raw)
	if errPrepare != nil {
		t.Fatalf("prepare replay: %v", errPrepare)
	}
	if !bytes.Equal(got, raw) {
		t.Fatalf("request without previous_response_id must remain unchanged: %s", got)
	}
}

func TestResponsesHTTPReplayPreservesEarlierToolOutputAcrossTurns(t *testing.T) {
	resetResponsesHTTPReplayCacheForTest()
	t.Cleanup(resetResponsesHTTPReplayCacheForTest)

	firstRequest := []byte(`{"model":"gpt-test","input":[{"role":"user","content":"start"}]}`)
	firstResponse := []byte(`{"id":"resp_spawn","output":[{"type":"function_call","call_id":"spawn_1","name":"spawn_agent","arguments":"{}"}]}`)
	rememberResponsesHTTPReplay(firstRequest, firstResponse)

	secondRequest := []byte(`{"model":"gpt-test","previous_response_id":"resp_spawn","input":[{"type":"function_call_output","call_id":"spawn_1","output":"child-1"}]}`)
	secondReplay, errSecond := prepareResponsesHTTPReplay(secondRequest)
	if errSecond != nil {
		t.Fatalf("prepare second turn: %v", errSecond)
	}
	secondResponse := []byte(`{"id":"resp_wait","output":[{"type":"function_call","call_id":"wait_1","name":"wait_agent","arguments":"{}"}]}`)
	rememberResponsesHTTPReplay(secondReplay, secondResponse)

	thirdRequest := []byte(`{"model":"gpt-test","previous_response_id":"resp_wait","input":[{"type":"function_call_output","call_id":"wait_1","output":"WAIT_SENTINEL_42"}]}`)
	thirdReplay, errThird := prepareResponsesHTTPReplay(thirdRequest)
	if errThird != nil {
		t.Fatalf("prepare third turn: %v", errThird)
	}
	thirdResponse := []byte(`{"id":"resp_close","output":[{"type":"function_call","call_id":"close_1","name":"close_agent","arguments":"{}"}]}`)
	rememberResponsesHTTPReplay(thirdReplay, thirdResponse)

	fourthRequest := []byte(`{"model":"gpt-test","previous_response_id":"resp_close","input":[{"type":"function_call_output","call_id":"close_1","output":"closed"}]}`)
	fourthReplay, errFourth := prepareResponsesHTTPReplay(fourthRequest)
	if errFourth != nil {
		t.Fatalf("prepare fourth turn: %v", errFourth)
	}

	found := false
	for _, item := range gjson.GetBytes(fourthReplay, "input").Array() {
		if item.Get("type").String() == "function_call_output" && item.Get("call_id").String() == "wait_1" && item.Get("output").String() == "WAIT_SENTINEL_42" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("earlier wait_agent output must survive later continuation: %s", fourthReplay)
	}
}

func TestResponsesHTTPReplayStoreSurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "responses-http-replay.json")
	store := newResponsesHTTPReplayStore(path)
	store.put("resp_restart", []byte(`[{"role":"user","content":"persist me"}]`))

	restarted := newResponsesHTTPReplayStore(path)
	got, ok := restarted.get("resp_restart")
	if !ok {
		t.Fatal("persisted replay entry must survive store reconstruction")
	}
	if string(got) != `[{"role":"user","content":"persist me"}]` {
		t.Fatalf("unexpected restored replay input: %s", got)
	}
}

func TestResponsesHTTPReplayStorePrunesExpiredOrderEntries(t *testing.T) {
	store := newResponsesHTTPReplayStore("")
	store.entries["expired"] = responsesHTTPReplayEntry{
		input:     []byte(`[]`),
		expiresAt: time.Now().Add(-time.Minute),
	}
	store.order = []string{"expired"}

	store.put("fresh", []byte(`[]`))
	if _, ok := store.entries["expired"]; ok {
		t.Fatal("expired replay entry must be removed")
	}
	if len(store.order) != 1 || store.order[0] != "fresh" {
		t.Fatalf("replay order must be compacted with entries, got: %#v", store.order)
	}
}

func TestResponsesSSEFramerCapturesCompletedReplay(t *testing.T) {
	framer := &responsesSSEFramer{}
	var downstream bytes.Buffer

	framer.WriteChunk(&downstream, []byte("event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"blue\"}]}}\n\n"))
	framer.WriteChunk(&downstream, []byte("event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream\",\"output\":[]}}\n\n"))

	if framer.completedResponseID != "resp_stream" {
		t.Fatalf("unexpected completed response id: %q", framer.completedResponseID)
	}
	output := gjson.ParseBytes(framer.completedOutput)
	if !output.IsArray() || len(output.Array()) != 1 {
		t.Fatalf("expected repaired completed output, got: %s", framer.completedOutput)
	}
	if got := output.Array()[0].Get("content.0.text").String(); got != "blue" {
		t.Fatalf("unexpected captured output: %q", got)
	}
}
