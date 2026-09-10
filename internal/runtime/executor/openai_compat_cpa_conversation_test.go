package executor

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

func TestOpenAICompatWebModelConversationIDIsCallerScopedAndSpoofProof(t *testing.T) {
	var gotHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{
		OpenAICompatibility: []config.OpenAICompatibility{{Name: "webmodel"}},
	})

	call := func(callerScope, sessionID string) string {
		t.Helper()
		gotHeaders = nil
		auth := &cliproxyauth.Auth{
			Provider: "openai-compatibility",
			Attributes: map[string]string{
				"base_url":                    server.URL,
				"api_key":                     "test-key",
				"compat_name":                 "webmodel",
				"header:X-CPA-Conversation-ID": "$X-CPA-Conversation-ID",
			},
		}
		req := cliproxyexecutor.Request{
			Model:   "gpt-test",
			Payload: []byte(`{"messages":[{"role":"user","content":"hello"}]}`),
		}
		opts := cliproxyexecutor.Options{
			SourceFormat: sdktranslator.FormatOpenAI,
			Headers: http.Header{
				"Session-Id":            []string{sessionID},
				"X-CPA-Conversation-ID": []string{"spoofed-by-caller"},
			},
			Metadata: map[string]any{
				cliproxyexecutor.CallerScopeMetadataKey: callerScope,
			},
		}
		if _, err := executor.Execute(context.Background(), auth, req, opts); err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		return gotHeaders.Get("X-CPA-Conversation-ID")
	}

	first := call("caller-a", "shared-session")
	second := call("caller-a", "shared-session")
	otherCaller := call("caller-b", "shared-session")
	otherSession := call("caller-a", "other-session")

	if first == "" {
		t.Fatal("webmodel conversation id must be present for a scoped session")
	}
	if first == "spoofed-by-caller" {
		t.Fatal("caller-provided conversation id must not reach WebModel")
	}
	if second != first {
		t.Fatalf("same caller/session produced unstable conversation id: first=%q second=%q", first, second)
	}
	if otherCaller == first {
		t.Fatalf("different caller scopes shared conversation id %q", first)
	}
	if otherSession == first {
		t.Fatalf("different sessions shared conversation id %q", first)
	}
}

func TestOpenAICompatConversationIDIsWebModelOnlyAndRequiresCallerScope(t *testing.T) {
	var gotHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer server.Close()

	run := func(compatName string, metadata map[string]any) string {
		t.Helper()
		gotHeaders = nil
		executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{
			OpenAICompatibility: []config.OpenAICompatibility{{Name: compatName}},
		})
		auth := &cliproxyauth.Auth{
			Provider: "openai-compatibility",
			Attributes: map[string]string{
				"base_url":                    server.URL,
				"api_key":                     "test-key",
				"compat_name":                 compatName,
				"header:X-CPA-Conversation-ID": "$X-CPA-Conversation-ID",
			},
		}
		req := cliproxyexecutor.Request{
			Model:   "gpt-test",
			Payload: []byte(`{"messages":[{"role":"user","content":"hello"}]}`),
		}
		opts := cliproxyexecutor.Options{
			SourceFormat: sdktranslator.FormatOpenAI,
			Headers: http.Header{
				"Session-Id":            []string{"same-session"},
				"X-CPA-Conversation-ID": []string{"spoofed-by-caller"},
			},
			Metadata: metadata,
		}
		if _, err := executor.Execute(context.Background(), auth, req, opts); err != nil {
			t.Fatalf("Execute() error = %v", err)
		}
		return gotHeaders.Get("X-CPA-Conversation-ID")
	}

	if got := run("webmodel", nil); got != "" {
		t.Fatalf("webmodel request without caller scope leaked conversation id %q", got)
	}
	if got := run("other", map[string]any{cliproxyexecutor.CallerScopeMetadataKey: "caller-a"}); got != "" {
		t.Fatalf("non-webmodel compat received internal conversation id %q", got)
	}
}
