package executor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
)

const cpaWebModelConversationHeader = "X-CPA-Conversation-ID"

func (e *OpenAICompatExecutor) applyCPAWebModelConversationHeader(ctx context.Context, req *http.Request, auth *cliproxyauth.Auth, opts cliproxyexecutor.Options) {
	if req == nil {
		return
	}
	req.Header.Del(cpaWebModelConversationHeader)

	compat := e.resolveCompatConfig(auth)
	if compat == nil || !strings.EqualFold(strings.TrimSpace(compat.Name), "webmodel") {
		return
	}
	callerScope, _ := opts.Metadata[cliproxyexecutor.CallerScopeMetadataKey].(string)
	callerScope = strings.TrimSpace(callerScope)
	sessionID := util.SessionIDFromContext(ctx)
	if callerScope == "" || sessionID == "" {
		return
	}

	sum := sha256.Sum256([]byte("cpa:webmodel-conversation:v1\x00" + callerScope + "\x00" + sessionID))
	req.Header.Set(cpaWebModelConversationHeader, hex.EncodeToString(sum[:]))
}
