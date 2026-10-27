package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
)

type replayDeliveryExecutor struct {
	prematureResponsesStreamExecutor
	terminal string
}

func (*replayDeliveryExecutor) Execute(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{Payload: []byte(`{"id":"resp_delivery","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}]}`)}, nil
}

func (e *replayDeliveryExecutor) ExecuteStream(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	chunks := make(chan coreexecutor.StreamChunk, 2)
	chunks <- coreexecutor.StreamChunk{Payload: []byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n")}
	if e.terminal != "" {
		chunks <- coreexecutor.StreamChunk{Payload: []byte(fmt.Sprintf("event: response.%s\ndata: {\"type\":\"response.%s\",\"response\":{\"id\":\"resp_delivery\",\"status\":\"%s\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}]}}\n\n", e.terminal, e.terminal, e.terminal))}
	}
	close(chunks)
	return &coreexecutor.StreamResult{Chunks: chunks}, nil
}

func TestResponsesHTTPReplayRequiresSuccessfulDelivery(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tc := range []struct {
		name, terminal                                       string
		stream, failWrite, shortWrite, failFlush, wantReplay bool
	}{
		{name: "json_success", wantReplay: true},
		{name: "json_write_failure", failWrite: true},
		{name: "json_short_write", failWrite: true, shortWrite: true},
		{name: "stream_success", stream: true, terminal: "completed", wantReplay: true},
		{name: "stream_write_failure", stream: true, terminal: "completed", failWrite: true},
		{name: "stream_short_write", stream: true, terminal: "completed", failWrite: true, shortWrite: true},
		{name: "stream_flush_failure", stream: true, terminal: "completed", failFlush: true},
		{name: "stream_partial_eof", stream: true},
		{name: "stream_failed", stream: true, terminal: "failed"},
		{name: "stream_incomplete", stream: true, terminal: "incomplete"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const model = "replay-delivery-model"
			enableResponsesHTTPReplayForTest(t, model)
			resetResponsesHTTPReplayCacheForTest()
			t.Cleanup(resetResponsesHTTPReplayCacheForTest)
			executor := &replayDeliveryExecutor{terminal: tc.terminal}
			manager := coreauth.NewManager(nil, nil, nil)
			manager.RegisterExecutor(executor)
			auth := &coreauth.Auth{ID: "replay-delivery-auth", Provider: executor.Identifier(), Status: coreauth.StatusActive}
			if _, err := manager.Register(context.Background(), auth); err != nil {
				t.Fatal(err)
			}
			registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{ID: model}})
			t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })
			h := NewOpenAIResponsesAPIHandler(handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))
			router := gin.New()
			router.POST("/v1/responses", h.Responses)
			request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(fmt.Sprintf(`{"model":%q,"input":"question","stream":%t}`, model, tc.stream)))
			writer := &issue6332Writer{ResponseRecorder: httptest.NewRecorder(), cancel: func() {}, event: "resp_delivery", failWrite: tc.failWrite, shortWrite: tc.shortWrite, failFlush: tc.failFlush}
			router.ServeHTTP(writer, request)
			if _, ok := responsesHTTPReplayCache.get("resp_delivery"); ok != tc.wantReplay {
				t.Fatalf("replay stored = %t, want %t; delivered body: %s", ok, tc.wantReplay, writer.Body.String())
			}
		})
	}
}
