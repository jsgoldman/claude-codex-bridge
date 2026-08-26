import { describe, expect, it } from "vitest";
import { buildClaudeInvocation } from "../src/lib/claude-invocation.js";

describe("buildClaudeInvocation", () => {
  it("sends the prompt through stdin so variadic tool options cannot consume it", () => {
    const prompt = "Review origin/main...HEAD without rubber-stamping.";

    expect(
      buildClaudeInvocation(prompt, {
        resumeSessionId: "session-123",
        model: "opus",
        maxTurns: 5,
        maxBudgetUsd: 12.5,
        disableMcpServers: true,
        tools: ["Read", "Bash"],
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
        "--max-budget-usd",
        "12.5",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--tools",
        "Read,Bash",
        "--allowedTools",
        "Read",
        "--allowedTools",
        "Bash(git diff *)",
      ],
      stdin: prompt,
    });
  });

  it("starts a fresh session when no explicit continuation token is provided", () => {
    const invocation = buildClaudeInvocation("Independent task", {
      maxTurns: 50,
    });

    expect(invocation.args).not.toContain("--resume");
  });
});
