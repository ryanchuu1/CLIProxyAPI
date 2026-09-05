package responses

import "testing"

func TestValidateOpenAIResponsesNamespaceToolsFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		raw  string
	}{
		{name: "empty namespace name", raw: `{"tools":[{"type":"namespace","name":"","tools":[{"type":"function","name":"lookup"}]}]}`},
		{name: "empty namespace tools", raw: `{"tools":[{"type":"namespace","name":"example","tools":[]}]}`},
		{name: "empty child name", raw: `{"tools":[{"type":"namespace","name":"example","tools":[{"type":"function","name":""}]}]}`},
		{name: "prequalified child", raw: `{"tools":[{"type":"namespace","name":"example","tools":[{"type":"function","name":"example__lookup"}]}]}`},
		{name: "flattened collision", raw: `{"tools":[{"type":"function","name":"example__lookup"},{"type":"namespace","name":"example","tools":[{"type":"function","name":"lookup"}]}]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateOpenAIResponsesNamespaceTools([]byte(tt.raw)); err == nil {
				t.Fatalf("invalid namespace declaration must fail closed: %s", tt.raw)
			}
		})
	}
}

func TestValidateOpenAIResponsesNamespaceToolsAllowsValidNamespace(t *testing.T) {
	raw := []byte(`{"tools":[{"type":"namespace","name":"example","tools":[{"type":"function","name":"lookup"}]}]}`)
	if err := ValidateOpenAIResponsesNamespaceTools(raw); err != nil {
		t.Fatalf("valid namespace declaration rejected: %v", err)
	}
}
