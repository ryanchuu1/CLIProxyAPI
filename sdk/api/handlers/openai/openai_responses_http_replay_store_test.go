package openai

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tidwall/gjson"
)

func TestResponsesHTTPReplayStoreKeepsOnlyPerTurnDelta(t *testing.T) {
	store, err := newResponsesHTTPReplayStore("", 24*time.Hour, 0)
	if err != nil {
		t.Fatalf("new replay store: %v", err)
	}

	if errPut := store.put(
		"resp_1",
		"",
		[]byte(`[{"role":"user","content":"first"}]`),
		[]byte(`[{"type":"message","role":"assistant","content":"one"}]`),
	); errPut != nil {
		t.Fatalf("put first turn: %v", errPut)
	}
	if errPut := store.put(
		"resp_2",
		"resp_1",
		[]byte(`[{"role":"user","content":"second"}]`),
		[]byte(`[{"type":"message","role":"assistant","content":"two"}]`),
	); errPut != nil {
		t.Fatalf("put second turn: %v", errPut)
	}

	entry := store.entries["resp_2"]
	if entry.parentResponseID != "resp_1" {
		t.Fatalf("parent response = %q, want resp_1", entry.parentResponseID)
	}
	if got := len(gjson.ParseBytes(entry.input).Array()); got != 1 {
		t.Fatalf("stored second-turn input must contain only the incremental turn, got %d items", got)
	}

	replayed, ok := store.get("resp_2")
	if !ok {
		t.Fatal("expected resp_2 to be replayable")
	}
	if got := len(gjson.ParseBytes(replayed).Array()); got != 4 {
		t.Fatalf("reconstructed transcript = %d items, want 4: %s", got, replayed)
	}
}

func TestResponsesHTTPReplayStorePersistsAcrossRestart(t *testing.T) {
	stateDir := filepath.Join(t.TempDir(), "responses-replay")
	store, err := newResponsesHTTPReplayStore(stateDir, 24*time.Hour, 0)
	if err != nil {
		t.Fatalf("new replay store: %v", err)
	}
	if errPut := store.put(
		"resp_1",
		"",
		[]byte(`[{"role":"user","content":"first"}]`),
		[]byte(`[{"type":"message","role":"assistant","content":"one"}]`),
	); errPut != nil {
		t.Fatalf("put first persisted turn: %v", errPut)
	}
	if errPut := store.put(
		"resp_2",
		"resp_1",
		[]byte(`[{"role":"user","content":"second"}]`),
		[]byte(`[{"type":"message","role":"assistant","content":"two"}]`),
	); errPut != nil {
		t.Fatalf("put second persisted turn: %v", errPut)
	}

	reloaded, errReload := newResponsesHTTPReplayStore(stateDir, 24*time.Hour, 0)
	if errReload != nil {
		t.Fatalf("reload replay store: %v", errReload)
	}
	replayed, ok := reloaded.get("resp_2")
	if !ok {
		t.Fatal("persisted response chain must survive store reconstruction")
	}
	items := gjson.ParseBytes(replayed).Array()
	if len(items) != 4 {
		t.Fatalf("reloaded transcript = %d items, want 4: %s", len(items), replayed)
	}
}

func TestResponsesHTTPReplayStoreSupportsBranches(t *testing.T) {
	store, err := newResponsesHTTPReplayStore("", 24*time.Hour, 0)
	if err != nil {
		t.Fatalf("new replay store: %v", err)
	}
	rootInput := []byte(`[{"role":"user","content":"root"}]`)
	rootOutput := []byte(`[{"type":"message","role":"assistant","content":"root answer"}]`)
	if errPut := store.put("resp_root", "", rootInput, rootOutput); errPut != nil {
		t.Fatalf("put root: %v", errPut)
	}
	if errPut := store.put("resp_left", "resp_root", []byte(`[{"role":"user","content":"left"}]`), []byte(`[{"type":"message","role":"assistant","content":"L"}]`)); errPut != nil {
		t.Fatalf("put left branch: %v", errPut)
	}
	if errPut := store.put("resp_right", "resp_root", []byte(`[{"role":"user","content":"right"}]`), []byte(`[{"type":"message","role":"assistant","content":"R"}]`)); errPut != nil {
		t.Fatalf("put right branch: %v", errPut)
	}

	left, okLeft := store.get("resp_left")
	right, okRight := store.get("resp_right")
	if !okLeft || !okRight {
		t.Fatal("both branches must remain replayable")
	}
	if strings.Contains(string(left), `"right"`) || strings.Contains(string(right), `"left"`) {
		t.Fatalf("branch transcripts must not bleed together: left=%s right=%s", left, right)
	}
}

func TestResponsesHTTPReplayStoreExpiresWholeConversation(t *testing.T) {
	store, err := newResponsesHTTPReplayStore("", time.Hour, 0)
	if err != nil {
		t.Fatalf("new replay store: %v", err)
	}
	now := time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC)
	store.now = func() time.Time { return now }
	if errPut := store.put("resp_root", "", []byte(`[]`), []byte(`[]`)); errPut != nil {
		t.Fatalf("put root: %v", errPut)
	}
	now = now.Add(30 * time.Minute)
	if errPut := store.put("resp_child", "resp_root", []byte(`[]`), []byte(`[]`)); errPut != nil {
		t.Fatalf("put child: %v", errPut)
	}

	now = now.Add(61 * time.Minute)
	if _, ok := store.get("resp_child"); ok {
		t.Fatal("inactive conversation must expire as a whole")
	}
	if _, rootExists := store.entries["resp_root"]; rootExists {
		t.Fatal("expired root must be removed with its descendants")
	}
	if _, childExists := store.entries["resp_child"]; childExists {
		t.Fatal("expired child must be removed with its root")
	}
}

func TestResponsesHTTPReplayStoreBudgetEvictsWholeOldConversation(t *testing.T) {
	store, err := newResponsesHTTPReplayStore("", 24*time.Hour, 220)
	if err != nil {
		t.Fatalf("new replay store: %v", err)
	}
	now := time.Date(2026, 8, 26, 0, 0, 0, 0, time.UTC)
	store.now = func() time.Time { return now }
	if errPut := store.put("old_root", "", []byte(`[{"role":"user","content":"old old old old old"}]`), []byte(`[{"type":"message","content":"old old old"}]`)); errPut != nil {
		t.Fatalf("put old root: %v", errPut)
	}
	now = now.Add(time.Minute)
	if errPut := store.put("old_child", "old_root", []byte(`[{"role":"user","content":"old child old child"}]`), []byte(`[{"type":"message","content":"old child"}]`)); errPut != nil {
		t.Fatalf("put old child: %v", errPut)
	}
	now = now.Add(time.Minute)
	if errPut := store.put("new_root", "", []byte(`[{"role":"user","content":"new new new new new"}]`), []byte(`[{"type":"message","content":"new new new"}]`)); errPut != nil {
		t.Fatalf("put new root: %v", errPut)
	}

	if _, ok := store.get("old_root"); ok {
		t.Fatal("old inactive conversation should be evicted before the active one")
	}
	if _, ok := store.get("old_child"); ok {
		t.Fatal("old conversation descendants must be evicted with their root")
	}
	if _, ok := store.get("new_root"); !ok {
		t.Fatal("new active conversation must not be truncated by the storage budget")
	}
}

func TestResponsesHTTPReplayStoreReloadKeepsNewestConversationOverBudget(t *testing.T) {
	stateDir := filepath.Join(t.TempDir(), "responses-replay")
	store, err := newResponsesHTTPReplayStore(stateDir, 24*time.Hour, 0)
	if err != nil {
		t.Fatalf("new replay store: %v", err)
	}
	if errPut := store.put("resp_large", "", []byte(`[{"role":"user","content":"this active conversation intentionally exceeds a tiny configured budget"}]`), []byte(`[{"type":"message","content":"still keep me"}]`)); errPut != nil {
		t.Fatalf("put persisted turn: %v", errPut)
	}

	reloaded, errReload := newResponsesHTTPReplayStore(stateDir, 24*time.Hour, 1)
	if errReload != nil {
		t.Fatalf("reload over-budget replay store: %v", errReload)
	}
	if _, ok := reloaded.get("resp_large"); !ok {
		t.Fatal("newest conversation must survive restart even when it alone exceeds the storage budget")
	}
}

func TestResponsesHTTPReplayStoreRejectsCorruptPersistentState(t *testing.T) {
	stateDir := filepath.Join(t.TempDir(), "responses-replay")
	if err := os.Mkdir(stateDir, 0o700); err != nil {
		t.Fatalf("mkdir state dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(stateDir, "corrupt.json"), []byte(`not-json`), 0o600); err != nil {
		t.Fatalf("write corrupt state: %v", err)
	}
	if _, err := newResponsesHTTPReplayStore(stateDir, 24*time.Hour, 0); err == nil {
		t.Fatal("corrupt persisted continuation state must fail closed")
	}
}
