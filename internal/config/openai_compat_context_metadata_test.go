package config

import (
	"testing"

	"gopkg.in/yaml.v3"
)

func TestOpenAICompatibilityModelContextLengthYAML(t *testing.T) {
	var model OpenAICompatibilityModel
	if err := yaml.Unmarshal([]byte("name: gemini-web/gemini-3.1-pro\nalias: webmodel-gemini-3.1-pro\ncontext-length: 1048576\n"), &model); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if model.ContextLength != 1_048_576 {
		t.Fatalf("expected context length 1048576, got %d", model.ContextLength)
	}
}
