package cliproxy

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
)

func TestBuildOpenAICompatibilityConfigModelsPreservesContextLength(t *testing.T) {
	models := buildOpenAICompatibilityConfigModels(&config.OpenAICompatibility{
		Name: "webmodel",
		Models: []config.OpenAICompatibilityModel{
			{Name: "gemini-web/gemini-3.1-pro", Alias: "webmodel-gemini-3.1-pro", ContextLength: 1_048_576},
			{Name: "unknown/model", Alias: "webmodel-unknown"},
		},
	})
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if got := models[0].ContextLength; got != 1_048_576 {
		t.Fatalf("expected context length 1048576, got %d", got)
	}
	if got := models[1].ContextLength; got != 0 {
		t.Fatalf("unknown context must remain zero, got %d", got)
	}
}
