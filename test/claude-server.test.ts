import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { describe, expect, it, vi } from "vitest";
import {
  createClaudeServer,
  precomputedReviewContext,
  truncateReviewSection,
  type ClaudeRunner,
  type ClaudeRunOptions,
  type ReviewContextResolver,
  type WorkspaceIdentityResolver,
} from "../src/claude-server.js";
import type { ClaudeResult } from "../src/lib/types.js";

function claudeResult(overrides: Partial<ClaudeResult> = {}): ClaudeResult {
  return {
    resultText: "done",
    sessionId: "session-returned",
    numTurns: 1,
    subtype: "success",
    isError: false,
    costUsd: 0.25,
    errors: [],
    ...overrides,
  };
}

async function connectBridge(
  runner: ClaudeRunner,
  workspaceIdentityResolver?: WorkspaceIdentityResolver,
  reviewContextResolver?: ReviewContextResolver,
) {
  const server = createClaudeServer(runner, workspaceIdentityResolver, reviewContextResolver);
  const client = new Client({ name: "claude-bridge-test", version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  return { client, server };
}

describe("Claude MCP tool contract", () => {
  it("never exceeds a review-section limit smaller than its truncation marker", () => {
    expect(truncateReviewSection("abcdef", 0, "[marker]")).toBe("");
    expect(truncateReviewSection("abcdef", 2, "[marker]")).toHaveLength(2);
    expect(truncateReviewSection("abcdef", 5, "[marker]").length).toBeLessThanOrEqual(5);
  });

  it("starts independent calls fresh and resumes only through a task-scoped continuation token", async () => {
    const calls: Array<{ prompt: string; options: ClaudeRunOptions }> = [];
    const runner: ClaudeRunner = vi.fn(async (prompt, options) => {
      calls.push({ prompt, options });
      if (calls.length === 3) return claudeResult({ sessionId: "returned-3" });
      return claudeResult({
        resultText: "partial review",
        sessionId: `returned-${calls.length}`,
        numTurns: 50,
        subtype: "error_max_turns",
        isError: true,
        errors: ["Reached maximum number of turns (50)"],
      });
    });
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({ name: "claude_query", arguments: { prompt: "first task" } });
      const second = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD~1..HEAD", workingDirectory: process.cwd() },
      });
      const continuationToken = (second.structuredContent as Record<string, unknown>)[
        "continuation_token"
      ];
      expect(continuationToken).toEqual(expect.any(String));
      expect(continuationToken).not.toBe("returned-2");

      await client.callTool({
        name: "claude_review_code",
        arguments: {
          target: "HEAD~1..HEAD",
          workingDirectory: process.cwd(),
          continuationToken,
          maxBudgetUsd: 7.5,
        },
      });

      expect(calls.map(({ options }) => options.resumeSessionId)).toEqual([
        undefined,
        undefined,
        "returned-2",
      ]);
      expect(calls[2]?.options.maxTurns).toBe(50);
      expect(calls[2]?.options.maxBudgetUsd).toBe(7.5);
      expect(calls[2]?.prompt).toContain("Continue the same task");
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("exposes terminal metadata as structured output", async () => {
    const runner: ClaudeRunner = async () =>
      claudeResult({
        resultText: "review complete",
        sessionId: "s-success",
        numTurns: 4,
        costUsd: 1.75,
      });
    const { client, server } = await connectBridge(runner);

    try {
      const response = await client.callTool({
        name: "claude_query",
        arguments: { prompt: "review" },
      });

      expect(response.isError).not.toBe(true);
      expect(response.structuredContent).toEqual({
        result: "review complete",
        session_id: "s-success",
        num_turns: 4,
        subtype: "success",
        is_error: false,
        cost: 1.75,
        errors: [],
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("applies the response limit to text and structured output", async () => {
    const runner: ClaudeRunner = async () =>
      claudeResult({ resultText: "x".repeat(90_000), sessionId: "s-large" });
    const { client, server } = await connectBridge(runner);

    try {
      const response = await client.callTool({
        name: "claude_query",
        arguments: { prompt: "large result" },
      });
      const text = (response.content[0] as { text: string }).text;
      expect(text).toHaveLength(80_025);
      expect(text).toMatch(/\.\.\.\[response truncated\]$/);
      expect(response.structuredContent?.result).toBe(text);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("keeps a complete result successful when only nonfatal diagnostics are present", async () => {
    const runner: ClaudeRunner = async () =>
      claudeResult({ resultText: "complete answer", isError: false, errors: ["diagnostic"] });
    const { client, server } = await connectBridge(runner);

    try {
      const response = await client.callTool({
        name: "claude_query",
        arguments: { prompt: "result with diagnostic" },
      });
      expect(response.isError).not.toBe(true);
      expect(response.structuredContent).toMatchObject({
        result: "complete answer",
        is_error: false,
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("returns max-turn exhaustion with partial output and an opaque continuation token", async () => {
    const runner: ClaudeRunner = async () =>
      claudeResult({
        resultText: "partial findings",
        sessionId: "s-max-turns",
        numTurns: 50,
        subtype: "error_max_turns",
        isError: true,
        costUsd: 3.5,
        errors: ["Reached maximum number of turns (50)"],
      });
    const { client, server } = await connectBridge(runner);

    try {
      const response = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD~1..HEAD" },
      });

      expect(response.isError).toBe(true);
      expect(response.structuredContent).toEqual({
        result: "partial findings",
        session_id: "s-max-turns",
        num_turns: 50,
        subtype: "error_max_turns",
        is_error: true,
        cost: 3.5,
        errors: ["Reached maximum number of turns (50)"],
        continuation_token: expect.any(String),
      });
      expect(response.content).toEqual([
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("partial findings"),
        }),
      ]);
      expect((response.content[0] as { text: string }).text).not.toContain("s-max-turns");
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("rejects a continuation token reused for another task, tool, or worktree", async () => {
    const calls: ClaudeRunOptions[] = [];
    const runner: ClaudeRunner = async (_prompt, options) => {
      calls.push(options);
      return claudeResult({
        resultText: "partial",
        sessionId: "scope-bound-session",
        subtype: "error_max_turns",
        isError: true,
      });
    };
    const { client, server } = await connectBridge(runner);

    try {
      const first = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd() },
      });
      const continuationToken = (first.structuredContent as Record<string, unknown>)[
        "continuation_token"
      ];

      const invalidCalls = [
        client.callTool({
          name: "claude_review_code",
          arguments: { target: "origin/main", workingDirectory: process.cwd(), continuationToken },
        }),
        client.callTool({
          name: "claude_review_plan",
          arguments: { plan: "HEAD", workingDirectory: process.cwd(), continuationToken },
        }),
        client.callTool({
          name: "claude_review_code",
          arguments: { target: "HEAD", workingDirectory: "/tmp", continuationToken },
        }),
      ];

      for (const invalidCall of invalidCalls) {
        const response = await invalidCall;
        expect(response.isError).toBe(true);
        expect(response.structuredContent).toMatchObject({
          subtype: "error_invalid_continuation",
          is_error: true,
        });
      }
      expect(calls).toHaveLength(1);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("keeps the same continuation token after a transient resume failure", async () => {
    const calls: ClaudeRunOptions[] = [];
    const runner: ClaudeRunner = async (_prompt, options) => {
      calls.push(options);
      if (calls.length === 1) {
        return claudeResult({
          resultText: "partial",
          sessionId: "durable-session",
          subtype: "error_max_turns",
          isError: true,
        });
      }
      if (calls.length === 2) {
        return claudeResult({
          resultText: "",
          sessionId: null,
          subtype: "error_timeout",
          isError: true,
          errors: ["Claude timed out"],
        });
      }
      return claudeResult({ resultText: "complete", sessionId: "durable-session" });
    };
    const { client, server } = await connectBridge(runner);

    try {
      const first = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd() },
      });
      const continuationToken = first.structuredContent?.continuation_token;

      const transientFailure = await client.callTool({
        name: "claude_review_code",
        arguments: {
          target: "HEAD",
          workingDirectory: process.cwd(),
          continuationToken,
        },
      });
      expect(transientFailure.structuredContent).toMatchObject({
        subtype: "error_timeout",
        is_error: true,
        continuation_token: continuationToken,
      });

      const completed = await client.callTool({
        name: "claude_review_code",
        arguments: {
          target: "HEAD",
          workingDirectory: process.cwd(),
          continuationToken,
        },
      });
      expect(completed.isError).not.toBe(true);
      expect(calls.map(({ resumeSessionId }) => resumeSessionId)).toEqual([
        undefined,
        "durable-session",
        "durable-session",
      ]);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("issues and preserves a task-scoped continuation after fresh and resumed budget exhaustion", async () => {
    const runner: ClaudeRunner = async () =>
      claudeResult({
        resultText: "partial budget-limited work",
        sessionId: "budget-session",
        numTurns: 12,
        subtype: "error_max_budget_usd",
        isError: true,
        costUsd: 4,
        errors: ["Reached maximum budget of $4"],
      });
    const { client, server } = await connectBridge(runner);

    try {
      const first = await client.callTool({
        name: "claude_review_plan",
        arguments: { plan: "Review this plan", workingDirectory: process.cwd() },
      });
      const continuationToken = first.structuredContent?.continuation_token;

      expect(continuationToken).toEqual(expect.any(String));
      expect((first.content[0] as { text: string }).text).toContain(
        "Claude reached the configured maximum budget.",
      );

      const resumed = await client.callTool({
        name: "claude_review_plan",
        arguments: {
          plan: "Review this plan",
          workingDirectory: process.cwd(),
          continuationToken,
        },
      });
      expect(resumed.structuredContent).toMatchObject({
        subtype: "error_max_budget_usd",
        is_error: true,
        errors: ["Reached maximum budget of $4"],
        continuation_token: continuationToken,
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("retires a continuation after a successful result without a subtype", async () => {
    const runner = vi.fn<ClaudeRunner>(async (_prompt, options) => {
      if (!options.resumeSessionId) {
        return claudeResult({
          resultText: "partial",
          sessionId: "raw-success-session",
          subtype: "error_max_turns",
          isError: true,
        });
      }
      return claudeResult({
        resultText: "completed raw result",
        sessionId: "raw-success-session",
        subtype: null,
        isError: false,
      });
    });
    const { client, server } = await connectBridge(runner);

    try {
      const first = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd() },
      });
      const continuationToken = first.structuredContent?.continuation_token;
      const completed = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd(), continuationToken },
      });

      expect(completed.isError).not.toBe(true);
      expect(completed.structuredContent).not.toHaveProperty("continuation_token");

      const reused = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd(), continuationToken },
      });
      expect(reused.structuredContent).toMatchObject({
        subtype: "error_invalid_continuation",
        is_error: true,
      });
      expect(runner).toHaveBeenCalledTimes(2);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("rejects concurrent reuse of one continuation token", async () => {
    let callCount = 0;
    let notifyResumeStarted!: () => void;
    let finishResume!: () => void;
    const resumeStarted = new Promise<void>((resolve) => {
      notifyResumeStarted = resolve;
    });
    const resumeResult = new Promise<ClaudeResult>((resolve) => {
      finishResume = () => resolve(claudeResult({ sessionId: "concurrent-session" }));
    });
    const runner: ClaudeRunner = async () => {
      callCount += 1;
      if (callCount === 1) {
        return claudeResult({
          resultText: "partial",
          sessionId: "concurrent-session",
          subtype: "error_max_turns",
          isError: true,
        });
      }
      notifyResumeStarted();
      return resumeResult;
    };
    const { client, server } = await connectBridge(runner);

    try {
      const first = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd() },
      });
      const continuationToken = first.structuredContent?.continuation_token;
      const activeResume = client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd(), continuationToken },
      });
      await resumeStarted;

      const concurrentResume = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: process.cwd(), continuationToken },
      });
      expect(concurrentResume.isError).toBe(true);
      expect(concurrentResume.structuredContent).toMatchObject({
        subtype: "error_continuation_in_use",
        is_error: true,
      });
      expect(callCount).toBe(2);

      finishResume();
      await activeResume;
    } finally {
      finishResume();
      await client.close();
      await server.close();
    }
  });

  it("rejects a continuation when only the branch identity changes", async () => {
    let branch = "feature-a";
    const resolveWorkspaceIdentity: WorkspaceIdentityResolver = async () => ["/worktree", branch];
    const runner = vi.fn<ClaudeRunner>(async () =>
      claudeResult({
        resultText: "partial",
        sessionId: "branch-session",
        subtype: "error_max_turns",
        isError: true,
      }),
    );
    const { client, server } = await connectBridge(runner, resolveWorkspaceIdentity);

    try {
      const first = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: "/same/path" },
      });
      const continuationToken = first.structuredContent?.continuation_token;
      branch = "feature-b";

      const response = await client.callTool({
        name: "claude_review_code",
        arguments: {
          target: "HEAD",
          workingDirectory: "/same/path",
          continuationToken,
        },
      });
      expect(response.isError).toBe(true);
      expect(response.structuredContent).toMatchObject({
        subtype: "error_invalid_continuation",
        is_error: true,
      });
      expect(runner).toHaveBeenCalledTimes(1);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("resolves workspace identity only when creating or consuming a continuation", async () => {
    let callCount = 0;
    const runner: ClaudeRunner = async () => {
      callCount += 1;
      if (callCount === 2) {
        return claudeResult({
          resultText: "partial",
          sessionId: "lazy-session",
          subtype: "error_max_turns",
          isError: true,
        });
      }
      return claudeResult();
    };
    const resolveWorkspaceIdentity = vi.fn<WorkspaceIdentityResolver>(async () => [
      "/worktree",
      "feature",
    ]);
    const { client, server } = await connectBridge(runner, resolveWorkspaceIdentity);

    try {
      await client.callTool({ name: "claude_query", arguments: { prompt: "fresh success" } });
      expect(resolveWorkspaceIdentity).not.toHaveBeenCalled();

      const exhausted = await client.callTool({
        name: "claude_query",
        arguments: { prompt: "resumable" },
      });
      expect(resolveWorkspaceIdentity).toHaveBeenCalledTimes(1);

      await client.callTool({
        name: "claude_query",
        arguments: {
          prompt: "resumable",
          continuationToken: exhausted.structuredContent?.continuation_token,
        },
      });
      expect(resolveWorkspaceIdentity).toHaveBeenCalledTimes(2);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("rejects a detached-worktree continuation after HEAD advances", async () => {
    const tempDirectory = mkdtempSync(join(tmpdir(), "ccb-detached-scope-"));
    execFileSync("git", ["init", "--quiet"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.email", "test@example.com"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.name", "Bridge Test"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "first\n");
    execFileSync("git", ["add", "tracked.txt"], { cwd: tempDirectory });
    execFileSync("git", ["commit", "--quiet", "-m", "first"], { cwd: tempDirectory });
    execFileSync("git", ["checkout", "--detach", "--quiet"], { cwd: tempDirectory });

    let callCount = 0;
    const runner: ClaudeRunner = async () => {
      callCount += 1;
      if (callCount === 1) {
        return claudeResult({
          resultText: "partial",
          sessionId: "detached-session",
          subtype: "error_max_turns",
          isError: true,
        });
      }
      return claudeResult({ sessionId: "detached-session" });
    };
    const { client, server } = await connectBridge(runner);

    try {
      const first = await client.callTool({
        name: "claude_implement",
        arguments: { task: "finish work", workingDirectory: tempDirectory },
      });
      writeFileSync(join(tempDirectory, "tracked.txt"), "second\n");
      execFileSync("git", ["add", "tracked.txt"], { cwd: tempDirectory });
      execFileSync("git", ["commit", "--quiet", "-m", "second"], { cwd: tempDirectory });

      const completed = await client.callTool({
        name: "claude_implement",
        arguments: {
          task: "finish work",
          workingDirectory: tempDirectory,
          continuationToken: first.structuredContent?.continuation_token,
        },
      });
      expect(completed.isError).toBe(true);
      expect(completed.structuredContent).toMatchObject({
        subtype: "error_invalid_continuation",
        is_error: true,
      });
      expect(callCount).toBe(1);
    } finally {
      await client.close();
      await server.close();
      rmSync(tempDirectory, { recursive: true, force: true });
    }
  });

  it("resumes an implementation with the exact session and full implementation tools", async () => {
    const calls: ClaudeRunOptions[] = [];
    const runner: ClaudeRunner = async (_prompt, options) => {
      calls.push(options);
      if (calls.length === 1) {
        return claudeResult({
          resultText: "partial implementation",
          sessionId: "implementation-session",
          subtype: "error_max_turns",
          isError: true,
        });
      }
      return claudeResult({ sessionId: "implementation-session" });
    };
    const { client, server } = await connectBridge(runner);

    try {
      const first = await client.callTool({
        name: "claude_implement",
        arguments: { task: "implement the fix", workingDirectory: process.cwd() },
      });
      await client.callTool({
        name: "claude_implement",
        arguments: {
          task: "implement the fix",
          workingDirectory: process.cwd(),
          continuationToken: first.structuredContent?.continuation_token,
        },
      });

      expect(calls[1]).toMatchObject({
        resumeSessionId: "implementation-session",
        maxTurns: 50,
        maxRetries: 0,
        disableMcpServers: true,
        tools: ["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
        allowedTools: ["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
      });
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("defaults review and implementation calls to 50 turns", async () => {
    const calls: Array<{ prompt: string; options: ClaudeRunOptions }> = [];
    const runner: ClaudeRunner = async (prompt, options) => {
      calls.push({ prompt, options });
      return claudeResult();
    };
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD~1..HEAD" },
      });
      await client.callTool({
        name: "claude_review_plan",
        arguments: { plan: "Do the work" },
      });
      await client.callTool({
        name: "claude_implement",
        arguments: { task: "Implement the work" },
      });

      expect(calls.map(({ options }) => options.maxTurns)).toEqual([50, 50, 50]);
      expect(calls.slice(0, 2).map(({ options }) => options.tools)).toEqual([
        ["Read", "Grep", "Glob"],
        ["Read", "Grep", "Glob"],
      ]);
      expect(calls[2]?.options.tools).toEqual(["Read", "Grep", "Glob", "Edit", "Write", "Bash"]);
      expect(calls[2]?.options.allowedTools).toEqual([
        "Read",
        "Grep",
        "Glob",
        "Edit",
        "Write",
        "Bash",
      ]);
      expect(calls.map(({ options }) => options.disableMcpServers)).toEqual([true, true, true]);
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("runs claude_query with the same isolated read-only policy as analysis tools", async () => {
    const calls: ClaudeRunOptions[] = [];
    const runner: ClaudeRunner = async (_prompt, options) => {
      calls.push(options);
      return claudeResult();
    };
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({
        name: "claude_query",
        arguments: { prompt: "Explain this repository" },
      });

      expect(calls).toHaveLength(1);
      expect(calls[0]).toMatchObject({
        settingSources: [],
        disableSlashCommands: true,
        permissionMode: "dontAsk",
        disableMcpServers: true,
        tools: ["Read", "Grep", "Glob"],
      });
      expect(calls[0]?.allowedTools).toBeUndefined();
      expect(calls[0]?.maxRetries).toBeUndefined();
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("advertises task-scoped continuation and budget controls on every Claude tool", async () => {
    const runner: ClaudeRunner = async () => claudeResult();
    const { client, server } = await connectBridge(runner);

    try {
      const { tools } = await client.listTools();
      expect(tools).toHaveLength(6);
      for (const tool of tools) {
        const properties = tool.inputSchema.properties as Record<string, unknown>;
        expect(properties).toHaveProperty("continuationToken");
        expect(properties).not.toHaveProperty("resumeSessionId");
        expect(properties).toHaveProperty("maxBudgetUsd");
        expect(tool.outputSchema).toBeDefined();
        expect(
          (tool.outputSchema as { properties: Record<string, unknown> }).properties,
        ).toHaveProperty("errors");
      }
      const queryProperties = tools.find(({ name }) => name === "claude_query")?.inputSchema
        .properties as Record<string, unknown>;
      expect(queryProperties).not.toHaveProperty("tools");
      expect(queryProperties).not.toHaveProperty("allowedTools");
      expect(tools.find(({ name }) => name === "claude_implement")?.description).toContain(
        "run shell commands without per-command prompts",
      );
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("rejects invalid turn limits before invoking Claude", async () => {
    const runner: ClaudeRunner = vi.fn(async () => claudeResult());
    const { client, server } = await connectBridge(runner);

    try {
      const zeroTurns = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", maxTurns: 0 },
      });
      const fractionalTurns = await client.callTool({
        name: "claude_implement",
        arguments: { task: "work", maxTurns: 2.5 },
      });
      expect(zeroTurns.isError).toBe(true);
      expect(fractionalTurns.isError).toBe(true);
      expect(runner).not.toHaveBeenCalled();
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("never interprets a review target as a git option", async () => {
    const tempDirectory = mkdtempSync(join(tmpdir(), "ccb-review-target-"));
    execFileSync("git", ["init", "--quiet"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.email", "test@example.com"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.name", "Bridge Test"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "before\n");
    execFileSync("git", ["add", "tracked.txt"], { cwd: tempDirectory });
    execFileSync("git", ["commit", "--quiet", "-m", "fixture"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "after\n");

    let prompt = "";
    const runner = vi.fn<ClaudeRunner>(async (receivedPrompt) => {
      prompt = receivedPrompt;
      return claudeResult();
    });
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({
        name: "claude_review_code",
        arguments: {
          target: "--stat",
          workingDirectory: tempDirectory,
        },
      });
      expect(prompt).not.toContain("Precomputed git diff");
      expect(prompt).toContain("Git diff was not attempted");
    } finally {
      await client.close();
      await server.close();
      rmSync(tempDirectory, { recursive: true, force: true });
    }
  });

  it("precomputes a review diff from an isolated fixture repository", async () => {
    const tempDirectory = mkdtempSync(join(tmpdir(), "ccb-review-diff-"));
    execFileSync("git", ["init", "--quiet"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.email", "test@example.com"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.name", "Bridge Test"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "before\n");
    execFileSync("git", ["add", "tracked.txt"], { cwd: tempDirectory });
    execFileSync("git", ["commit", "--quiet", "-m", "fixture"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "after\n");

    let prompt = "";
    const runner = vi.fn<ClaudeRunner>(async (receivedPrompt) => {
      prompt = receivedPrompt;
      return claudeResult();
    });
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: tempDirectory },
      });
      expect(prompt).toContain("Precomputed git diff (read-only bridge output)");
      expect(prompt).toContain("-before");
      expect(prompt).toContain("+after");
    } finally {
      await client.close();
      await server.close();
      rmSync(tempDirectory, { recursive: true, force: true });
    }
  });

  it("reports an empty review diff and lists untracked files for direct inspection", async () => {
    const tempDirectory = mkdtempSync(join(tmpdir(), "ccb-review-untracked-"));
    execFileSync("git", ["init", "--quiet"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.email", "test@example.com"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.name", "Bridge Test"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "unchanged\n");
    execFileSync("git", ["add", "tracked.txt"], { cwd: tempDirectory });
    execFileSync("git", ["commit", "--quiet", "-m", "fixture"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "new-test.ts"), "export const value = 1;\n");

    let prompt = "";
    const runner: ClaudeRunner = async (receivedPrompt) => {
      prompt = receivedPrompt;
      return claudeResult();
    };
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: tempDirectory },
      });
      expect(prompt).toContain("No git diff was produced for this target");
      expect(prompt).toContain(
        "Worktree-wide untracked files may be unrelated to the requested target",
      );
      expect(prompt).toContain("new-test.ts");
    } finally {
      await client.close();
      await server.close();
      rmSync(tempDirectory, { recursive: true, force: true });
    }
  });

  it("fails closed when a real repository diff exceeds the review limit", async () => {
    const tempDirectory = mkdtempSync(join(tmpdir(), "ccb-review-truncated-"));
    execFileSync("git", ["init", "--quiet"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.email", "test@example.com"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.name", "Bridge Test"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "before\n");
    execFileSync("git", ["add", "tracked.txt"], { cwd: tempDirectory });
    execFileSync("git", ["commit", "--quiet", "-m", "fixture"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "after\n".repeat(60_000));

    let prompt = "";
    const runner = vi.fn<ClaudeRunner>(async (receivedPrompt) => {
      prompt = receivedPrompt;
      return claudeResult();
    });
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: tempDirectory },
      });
      expect(prompt).toBe("");
      expect(runner).not.toHaveBeenCalled();
    } finally {
      await client.close();
      await server.close();
      rmSync(tempDirectory, { recursive: true, force: true });
    }
  });

  it("does not invoke Claude when a revision-range diff is truncated", async () => {
    const runner = vi.fn<ClaudeRunner>(async () => claudeResult());
    const resolveReviewContext = async () =>
      precomputedReviewContext("HEAD~1..HEAD", "/worktree", async (args) => {
        if (args[0] === "diff") return "diff line\n".repeat(30_000);
        return "";
      });
    const { client, server } = await connectBridge(
      runner,
      async () => ["/worktree", "feature"],
      resolveReviewContext,
    );

    try {
      const response = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD~1..HEAD", workingDirectory: "/worktree" },
      });
      expect(response.isError).toBe(true);
      expect(response.structuredContent).toMatchObject({
        subtype: "error_incomplete_review_evidence",
        is_error: true,
      });
      expect(runner).not.toHaveBeenCalled();
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("fails closed when a real repository diff exceeds the capture limit", async () => {
    const tempDirectory = mkdtempSync(join(tmpdir(), "ccb-review-limit-"));
    execFileSync("git", ["init", "--quiet"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.email", "test@example.com"], { cwd: tempDirectory });
    execFileSync("git", ["config", "user.name", "Bridge Test"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "before\n");
    execFileSync("git", ["add", "tracked.txt"], { cwd: tempDirectory });
    execFileSync("git", ["commit", "--quiet", "-m", "fixture"], { cwd: tempDirectory });
    writeFileSync(join(tempDirectory, "tracked.txt"), "after\n".repeat(400_000));

    let prompt = "";
    const runner = vi.fn<ClaudeRunner>(async (receivedPrompt) => {
      prompt = receivedPrompt;
      return claudeResult();
    });
    const { client, server } = await connectBridge(runner);

    try {
      await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: tempDirectory },
      });
      expect(prompt).toBe("");
      expect(runner).not.toHaveBeenCalled();
    } finally {
      await client.close();
      await server.close();
      rmSync(tempDirectory, { recursive: true, force: true });
    }
  });

  it("does not invoke Claude when revision-range diff capture exceeds the bridge limit", async () => {
    const outputLimitError = Object.assign(new RangeError("stdout maxBuffer length exceeded"), {
      code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
    });
    const runner = vi.fn<ClaudeRunner>(async () => claudeResult());
    const resolveReviewContext = async () =>
      precomputedReviewContext("HEAD~1..HEAD", "/worktree", async (args) => {
        if (args[0] === "diff") throw outputLimitError;
        return "";
      });
    const { client, server } = await connectBridge(
      runner,
      async () => ["/worktree", "feature"],
      resolveReviewContext,
    );

    try {
      const response = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD~1..HEAD", workingDirectory: "/worktree" },
      });
      expect(response.isError).toBe(true);
      expect(response.structuredContent?.errors).toEqual([
        "The git diff exceeded the bridge capture limit. Narrow the review target and retry.",
      ]);
      expect(runner).not.toHaveBeenCalled();
    } finally {
      await client.close();
      await server.close();
    }
  });

  it("reports an untracked-file capture limit even when the git diff succeeds", async () => {
    const outputLimitError = Object.assign(new RangeError("stdout maxBuffer length exceeded"), {
      code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
    });
    const context = await precomputedReviewContext("HEAD", "/worktree", async (args) => {
      if (args[0] === "diff") return "diff --git a/file.ts b/file.ts";
      throw outputLimitError;
    });

    expect(context).toContain("Precomputed git diff (read-only bridge output)");
    expect(context).toContain(
      "Worktree-wide untracked-file discovery exceeded the bridge's 2,000,000-byte capture limit",
    );
  });

  it("reports an untracked-file git failure instead of silently omitting the status", async () => {
    const context = await precomputedReviewContext("HEAD", "/worktree", async (args) => {
      if (args[0] === "diff") return "diff --git a/file.ts b/file.ts";
      throw Object.assign(new Error("not a repository"), { code: 128 });
    });

    expect(context).toContain("Precomputed git diff (read-only bridge output)");
    expect(context).toContain("Worktree-wide untracked-file discovery failed");
  });

  it("does not recompute review context when resuming an existing session", async () => {
    let callCount = 0;
    const runner: ClaudeRunner = async () => {
      callCount += 1;
      if (callCount === 1) {
        return claudeResult({
          resultText: "partial",
          sessionId: "review-context-session",
          subtype: "error_max_turns",
          isError: true,
        });
      }
      return claudeResult({ sessionId: "review-context-session" });
    };
    const resolveWorkspaceIdentity: WorkspaceIdentityResolver = async () => [
      "/worktree",
      "feature",
    ];
    const resolveReviewContext = vi.fn<ReviewContextResolver>(async () => "review context");
    const { client, server } = await connectBridge(
      runner,
      resolveWorkspaceIdentity,
      resolveReviewContext,
    );

    try {
      const first = await client.callTool({
        name: "claude_review_code",
        arguments: { target: "HEAD", workingDirectory: "/worktree" },
      });
      await client.callTool({
        name: "claude_review_code",
        arguments: {
          target: "HEAD",
          workingDirectory: "/worktree",
          continuationToken: first.structuredContent?.continuation_token,
        },
      });
      expect(resolveReviewContext).toHaveBeenCalledTimes(1);
    } finally {
      await client.close();
      await server.close();
    }
  });
});
