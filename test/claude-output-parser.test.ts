import { describe, it, expect } from "vitest";
import { parseClaudeOutput } from "../src/lib/claude-output-parser.js";

describe("parseClaudeOutput", () => {
  it("parses result as string", () => {
    const json = JSON.stringify({
      result: "The answer is 42.",
      session_id: "s-123",
    });

    const result = parseClaudeOutput(json);
    expect(result.resultText).toBe("The answer is 42.");
    expect(result.sessionId).toBe("s-123");
    expect(result.errors).toHaveLength(0);
  });

  it("parses result with content array", () => {
    const json = JSON.stringify({
      result: {
        content: [
          { type: "text", text: "Hello " },
          { type: "text", text: "World" },
        ],
      },
    });

    const result = parseClaudeOutput(json);
    expect(result.resultText).toBe("Hello \nWorld");
  });

  it("extracts the terminal result from Claude Code event-array output", () => {
    const json = JSON.stringify([
      { type: "system", subtype: "init", session_id: "s-array" },
      { type: "assistant", message: { content: [{ type: "text", text: "working" }] } },
      {
        type: "result",
        subtype: "success",
        is_error: false,
        result: "BRIDGE_OK",
        session_id: "s-array",
        num_turns: 3,
        total_cost_usd: 0.2,
      },
    ]);

    const result = parseClaudeOutput(json);
    expect(result.resultText).toBe("BRIDGE_OK");
    expect(result.sessionId).toBe("s-array");
    expect(result.numTurns).toBe(3);
    expect(result.subtype).toBe("success");
    expect(result.isError).toBe(false);
    expect(result.costUsd).toBe(0.2);
    expect(result.errors).toHaveLength(0);
  });

  it("extracts terminal errors from Claude Code event-array output", () => {
    const json = JSON.stringify([
      { type: "system", subtype: "init", session_id: "s-error" },
      {
        type: "result",
        subtype: "error_max_turns",
        is_error: true,
        errors: ["Reached maximum number of turns (1)"],
        session_id: "s-error",
        num_turns: 1,
        total_cost_usd: 0.42,
      },
    ]);

    const result = parseClaudeOutput(json);
    expect(result.resultText).toBe("");
    expect(result.sessionId).toBe("s-error");
    expect(result.numTurns).toBe(1);
    expect(result.subtype).toBe("error_max_turns");
    expect(result.isError).toBe(true);
    expect(result.costUsd).toBe(0.42);
    expect(result.errors).toEqual(["Reached maximum number of turns (1)"]);
  });

  it("falls back to raw text when JSON parsing fails", () => {
    const result = parseClaudeOutput("This is plain text output");
    expect(result.resultText).toBe("This is plain text output");
    expect(result.errors).toHaveLength(0);
  });

  it("handles empty output", () => {
    const result = parseClaudeOutput("");
    expect(result.resultText).toBe("");
    expect(result.errors).toContain("Empty output from Claude CLI");
  });

  it("extracts error field", () => {
    const json = JSON.stringify({
      error: "Authentication failed",
      result: "",
    });

    const result = parseClaudeOutput(json);
    expect(result.errors).toContain("Authentication failed");
    expect(result.isError).toBe(true);
  });

  it("extracts cost_usd", () => {
    const json = JSON.stringify({
      result: "Done",
      cost_usd: 0.05,
    });

    const result = parseClaudeOutput(json);
    expect(result.costUsd).toBe(0.05);
  });

  it("ignores a non-numeric cost without losing the result", () => {
    const result = parseClaudeOutput(
      JSON.stringify({ result: "Done", total_cost_usd: "not-a-number" }),
    );

    expect(result.resultText).toBe("Done");
    expect(result.costUsd).toBeNull();
    expect(result.errors).toHaveLength(0);
  });

  it("handles nested error object", () => {
    const json = JSON.stringify({
      error: { message: "Rate limit exceeded", code: 429 },
    });

    const result = parseClaudeOutput(json);
    expect(result.errors).toContain("Rate limit exceeded");
  });

  it("stringifies response when no result field found", () => {
    const json = JSON.stringify({ data: "some unexpected format" });

    const result = parseClaudeOutput(json);
    expect(result.resultText).toContain("some unexpected format");
  });
});
