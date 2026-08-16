import { describe, expect, it } from "vitest";
import { buildClaudeInvocation } from "../src/lib/claude-invocation.js";

describe("buildClaudeInvocation", () => {
  it("sends the prompt through stdin so variadic tool options cannot consume it", () => {
    const prompt = "Review origin/main...HEAD without rubber-stamping.";

    expect(
      buildClaudeInvocation(prompt, {
        sessionId: "session-123",
        model: "opus",
        maxTurns: 5,
        allowedTools: ["Read", "Bash(git diff *)"],
      }),
    ).toEqual({
      args: [
        "-p",
        "--output-format",
        "json",
        "--resume",
        "session-123",
        "--model",
        "opus",
        "--max-turns",
        "5",
        "--allowedTools",
        "Read",
        "--allowedTools",
        "Bash(git diff *)",
      ],
      stdin: prompt,
    });
  });
});
