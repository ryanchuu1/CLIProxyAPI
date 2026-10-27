package openai

import (
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
)

func validateOpenAIResponsesNamespaceTools(requestRawJSON []byte) error {
	if !gjson.ValidBytes(requestRawJSON) {
		return fmt.Errorf("invalid JSON payload")
	}
	tools := gjson.GetBytes(requestRawJSON, "tools")
	if !tools.Exists() {
		return nil
	}
	if !tools.IsArray() {
		return fmt.Errorf("tools must be an array")
	}
	seenNames := map[string]struct{}{}
	var validationErr error
	tools.ForEach(func(_, tool gjson.Result) bool {
		switch strings.TrimSpace(tool.Get("type").String()) {
		case "", "function", "custom":
			name := responsesHandlerToolName(tool)
			if name == "" {
				validationErr = fmt.Errorf("tool name must not be empty")
				return false
			}
			if err := recordResponsesHandlerToolName(seenNames, name); err != nil {
				validationErr = err
				return false
			}
		case "namespace":
			if err := validateResponsesHandlerNamespaceTool(seenNames, tool); err != nil {
				validationErr = err
				return false
			}
		}
		return true
	})
	return validationErr
}

func validateResponsesHandlerNamespaceTool(seenNames map[string]struct{}, tool gjson.Result) error {
	namespaceName := strings.TrimSpace(tool.Get("name").String())
	if namespaceName == "" {
		return fmt.Errorf("namespace tool name must not be empty")
	}
	if !strings.HasPrefix(namespaceName, "mcp__") && strings.Contains(namespaceName, "__") {
		return fmt.Errorf("namespace tool name must not contain __")
	}
	children := tool.Get("tools")
	if !children.Exists() || !children.IsArray() {
		return fmt.Errorf("namespace tool %q must contain a tools array", namespaceName)
	}
	if len(children.Array()) == 0 {
		return fmt.Errorf("namespace tool %q must contain at least one child tool", namespaceName)
	}
	childNames := map[string]struct{}{}
	var validationErr error
	children.ForEach(func(_, child gjson.Result) bool {
		childType := strings.TrimSpace(child.Get("type").String())
		if childType != "" && childType != "function" && childType != "custom" {
			return true
		}
		childName := responsesHandlerToolName(child)
		if childName == "" {
			validationErr = fmt.Errorf("namespace child tool name must not be empty")
			return false
		}
		if strings.HasPrefix(childName, namespaceName+"__") {
			validationErr = fmt.Errorf("namespace child tool name must not be pre-qualified")
			return false
		}
		if err := recordResponsesHandlerToolName(childNames, childName); err != nil {
			validationErr = fmt.Errorf("duplicate namespace child tool name %q", childName)
			return false
		}
		if err := recordResponsesHandlerToolName(seenNames, qualifyResponsesHandlerNamespaceToolName(namespaceName, childName)); err != nil {
			validationErr = err
			return false
		}
		return true
	})
	return validationErr
}

func recordResponsesHandlerToolName(seenNames map[string]struct{}, name string) error {
	if _, exists := seenNames[name]; exists {
		return fmt.Errorf("duplicate tool name %q", name)
	}
	seenNames[name] = struct{}{}
	return nil
}

func qualifyResponsesHandlerNamespaceToolName(namespaceName, childName string) string {
	return strings.TrimSpace(namespaceName) + "__" + strings.TrimSpace(childName)
}

func responsesHandlerToolName(tool gjson.Result) string {
	if name := strings.TrimSpace(tool.Get("name").String()); name != "" {
		return name
	}
	return strings.TrimSpace(tool.Get("function.name").String())
}
