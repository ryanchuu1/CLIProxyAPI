package openai

import (
	"bytes"
	"testing"

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

func TestResponsesHTTPReplayUnknownIDPreservesRequest(t *testing.T) {
	resetResponsesHTTPReplayCacheForTest()
	t.Cleanup(resetResponsesHTTPReplayCacheForTest)

	raw := []byte(`{"model":"gpt-test","previous_response_id":"resp_missing","input":"hello"}`)
	got, errPrepare := prepareResponsesHTTPReplay(raw)
	if errPrepare != nil {
		t.Fatalf("prepare replay: %v", errPrepare)
	}
	if !bytes.Equal(got, raw) {
		t.Fatalf("unknown response id must remain untouched: %s", got)
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
