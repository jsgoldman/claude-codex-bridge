#!/usr/bin/env node
import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { realpathSync } from "node:fs";
import { realpath } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { buildClaudeInvocation } from "./lib/claude-invocation.js";
import { execCommand } from "./lib/exec-runner.js";
import { parseClaudeOutput } from "./lib/claude-output-parser.js";
import { buildExplainCodePrompt, buildPlanPerfPrompt } from "./lib/prompt-builder.js";
import { createProgressReporter, logger, type ProgressReporter } from "./lib/logger.js";
import { CLAUDE_MODELS } from "./lib/types.js";
import type { ClaudeResult } from "./lib/types.js";

export interface ClaudeRunOptions {
  workingDirectory?: string;
  model?: string;
  maxTurns?: number;
  maxBudgetUsd?: number;
  disableMcpServers?: boolean;
  tools?: string[];
  allowedTools?: string[];
  resumeSessionId?: string;
  progress?: ProgressReporter;
}

export type ClaudeRunner = (prompt: string, options?: ClaudeRunOptions) => Promise<ClaudeResult>;

export type WorkspaceIdentityResolver = (
  workingDirectory?: string,
) => Promise<readonly [canonicalWorktree: string, revision: string]>;

export type ReviewContextResolver = (target: string, workingDirectory?: string) => Promise<string>;

export type GitOutputRunner = (args: string[], workingDirectory: string) => Promise<string>;

const execFileAsync = promisify(execFile);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export async function runClaude(
  prompt: string,
  options: ClaudeRunOptions = {},
): Promise<ClaudeResult> {
  const invocation = buildClaudeInvocation(prompt, options);

  options.progress?.report("Starting claude...");

  // Buffer for partial stderr lines split across chunks.
  let stderrBuf = "";

  const result = await execCommand({
    command: "claude",
    args: invocation.args,
    stdin: invocation.stdin,
    cwd: options.workingDirectory,
    onStdout: (chunk) => {
      logger.info(`[claude] ${chunk.toString().replace(/\n$/, "")}`);
    },
    onStderr: (chunk) => {
      const text = chunk.toString();
      logger.warn(`[claude:stderr] ${text.replace(/\n$/, "")}`);

      // Forward complete stderr lines as inline progress.
      if (options.progress) {
        stderrBuf += text;
        const lines = stderrBuf.split(/\r?\n|\r/);
        stderrBuf = lines.pop()!;
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed) {
            options.progress.report(trimmed);
          }
        }
      }
    },
  });

  // Flush any remaining buffered stderr fragment.
  if (options.progress && stderrBuf.trim()) {
    options.progress.report(stderrBuf.trim());
  }

  if (result.timedOut) {
    return {
      resultText: "",
      sessionId: null,
      numTurns: null,
      subtype: "error_timeout",
      isError: true,
      costUsd: null,
      errors: ["Claude timed out. Increase BRIDGE_TIMEOUT_MS if needed."],
    };
  }

  options.progress?.report("Parsing response...");
  const parsed = parseClaudeOutput(result.stdout);

  // Check stderr for API key issues
  if (result.exitCode !== 0 && !parsed.resultText) {
    const stderr = result.stderr.toLowerCase();
    if (
      stderr.includes("api key") ||
      stderr.includes("authentication") ||
      stderr.includes("unauthorized")
    ) {
      parsed.errors.push("Claude API key issue. Ensure ANTHROPIC_API_KEY is set.");
    } else if (result.stderr.trim()) {
      parsed.errors.push(result.stderr.trim());
    }
    parsed.isError = true;
  }

  return parsed;
}

// Safety limit to prevent exceeding MCP response token limits.
const MAX_RESPONSE_CHARS = 80_000;

export function formatClaudeResponse(
  parsed: ClaudeResult,
  continuationToken?: string,
): {
  content: Array<{ type: "text"; text: string }>;
  structuredContent: Record<string, unknown>;
  isError?: boolean;
} {
  const isMaxTurns = parsed.subtype === "error_max_turns";
  const resultText =
    parsed.resultText.length > MAX_RESPONSE_CHARS
      ? parsed.resultText.slice(0, MAX_RESPONSE_CHARS) + "\n\n...[response truncated]"
      : parsed.resultText;
  const isError =
    isMaxTurns || parsed.isError || (parsed.errors.length > 0 && parsed.resultText.length === 0);
  const structuredContent: Record<string, unknown> = {
    result: resultText,
    session_id: parsed.sessionId,
    num_turns: parsed.numTurns,
    subtype: parsed.subtype,
    is_error: isError,
    cost: parsed.costUsd,
  };
  if (continuationToken) {
    structuredContent["continuation_token"] = continuationToken;
  }

  let text = resultText;
  if (isMaxTurns) {
    const continuationText = continuationToken
      ? `Resume only this exact task by calling the same tool with continuationToken: "${continuationToken}".`
      : "Claude did not return a resumable session ID.";
    const maxTurnsText = `Claude reached the configured maximum number of turns. ${continuationText}`;
    text = text ? `${text}\n\n${maxTurnsText}` : maxTurnsText;
    if (parsed.errors.length > 0) text += `\n\n${parsed.errors.join("; ")}`;
  } else if (parsed.errors.length > 0) {
    const errorText = `Error: ${parsed.errors.join("; ")}`;
    text = text ? `${text}\n\n${errorText}` : errorText;
    if (continuationToken) {
      text += `\n\nRetry this exact task with continuationToken: "${continuationToken}".`;
    }
  }

  if (text.length > MAX_RESPONSE_CHARS) {
    text = text.slice(0, MAX_RESPONSE_CHARS) + "\n\n...[response truncated]";
  }

  return {
    content: [{ type: "text" as const, text }],
    structuredContent,
    ...(isError ? { isError: true } : {}),
  };
}

const READ_ONLY_AVAILABLE_TOOLS = ["Read", "Grep", "Glob"];

const READ_ONLY_ALLOWED_TOOLS = ["Read", "Grep", "Glob"];

const IMPLEMENTATION_AVAILABLE_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write", "Bash"];

const CLAUDE_OUTPUT_SCHEMA = {
  result: z.string(),
  session_id: z.string().nullable(),
  num_turns: z.number().int().nonnegative().nullable(),
  subtype: z.string().nullable(),
  is_error: z.boolean(),
  cost: z.number().nonnegative().nullable(),
  continuation_token: z.string().optional(),
};

const CONTINUATION_INPUT_SCHEMA = {
  continuationToken: z
    .string()
    .min(1)
    .optional()
    .describe("Opaque token returned by a resumable result for this exact task"),
  maxBudgetUsd: z
    .number()
    .positive()
    .optional()
    .describe("Maximum Claude API spend in US dollars for this call"),
};

interface TaskContinuation {
  scope: string;
  sessionId: string;
  expiresAt: number;
  inFlight: boolean;
}

interface ScopedClaudeTask {
  toolName: string;
  taskIdentity: readonly unknown[];
  prompt: string;
  workingDirectory?: string;
  continuationToken?: string;
  runOptions: Omit<ClaudeRunOptions, "workingDirectory" | "resumeSessionId">;
}

const CONTINUATION_PROMPT =
  "Continue the same task to completion. Use the existing session context and return the requested final result.";

const CONTINUATION_TTL_MS = 24 * 60 * 60 * 1_000;
const MAX_CONTINUATIONS = 1_000;
const MAX_REVIEW_DIFF_CHARS = 200_000;
const MAX_UNTRACKED_LIST_CHARS = 20_000;

async function gitOutput(args: string[], workingDirectory: string): Promise<string> {
  const { stdout } = await execFileAsync("git", args, {
    cwd: workingDirectory,
    encoding: "utf8",
    maxBuffer: 2_000_000,
  });
  return String(stdout).trim();
}

function isGitOutputLimitError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER"
  );
}

async function workspaceIdentity(workingDirectory?: string): Promise<readonly [string, string]> {
  const resolvedDirectory = resolve(workingDirectory ?? process.cwd());
  let canonicalDirectory = resolvedDirectory;
  try {
    canonicalDirectory = await realpath(resolvedDirectory);
  } catch {
    return [canonicalDirectory, "not-a-git-worktree"];
  }

  try {
    const repositoryRoot = await gitOutput(["rev-parse", "--show-toplevel"], canonicalDirectory);
    let revision: string;
    try {
      revision = await gitOutput(
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        canonicalDirectory,
      );
    } catch {
      revision = "detached";
    }
    return [await realpath(repositoryRoot), revision];
  } catch {
    return [canonicalDirectory, "not-a-git-worktree"];
  }
}

async function taskScope(
  task: ScopedClaudeTask,
  resolveWorkspaceIdentity: WorkspaceIdentityResolver,
): Promise<string> {
  return JSON.stringify([
    task.toolName,
    ...(await resolveWorkspaceIdentity(task.workingDirectory)),
    task.taskIdentity,
  ]);
}

function invalidContinuationResult(): ClaudeResult {
  return {
    resultText: "",
    sessionId: null,
    numTurns: null,
    subtype: "error_invalid_continuation",
    isError: true,
    costUsd: null,
    errors: ["Continuation token does not belong to this tool, task, branch, or worktree."],
  };
}

function continuationInUseResult(): ClaudeResult {
  return {
    resultText: "",
    sessionId: null,
    numTurns: null,
    subtype: "error_continuation_in_use",
    isError: true,
    costUsd: null,
    errors: [
      "Continuation token is already running. Wait for that call to finish before retrying.",
    ],
  };
}

export function truncateReviewSection(value: string, maxChars: number, marker: string): string {
  if (maxChars === Number.POSITIVE_INFINITY || value.length <= maxChars) return value;
  const limit = Number.isFinite(maxChars) ? Math.max(0, Math.floor(maxChars)) : 0;
  if (limit === 0) return "";
  const boundedMarker = marker.slice(0, limit);
  const lastContentIndex = limit - boundedMarker.length - 2;
  if (lastContentIndex <= 0) return boundedMarker;
  const lastLineBreak = value.lastIndexOf("\n", lastContentIndex);
  const truncationIndex = lastLineBreak > 0 ? lastLineBreak : lastContentIndex;
  return `${value.slice(0, truncationIndex)}\n\n${boundedMarker}`;
}

export async function precomputedReviewContext(
  target: string,
  workingDirectory?: string,
  runGitOutput: GitOutputRunner = gitOutput,
): Promise<string> {
  if (target.startsWith("-") || !/^[0-9A-Za-z_./~^{}:@-]+$/.test(target)) {
    return "Git diff was not attempted because the target cannot be passed as one safe Git revision or path. Inspect the target directly with Read, Grep, and Glob.";
  }

  const directory = resolve(workingDirectory ?? process.cwd());
  let diff = "";
  let diffFailure: "none" | "outputLimit" | "git" = "none";
  try {
    diff = await runGitOutput(
      ["diff", "--no-ext-diff", "--no-textconv", "--end-of-options", target],
      directory,
    );
  } catch (error) {
    diffFailure = isGitOutputLimitError(error) ? "outputLimit" : "git";
  }

  let untrackedFiles = "";
  let untrackedFailure: "none" | "outputLimit" | "git" = "none";
  try {
    untrackedFiles = await runGitOutput(["ls-files", "--others", "--exclude-standard"], directory);
  } catch (error) {
    untrackedFailure = isGitOutputLimitError(error) ? "outputLimit" : "git";
  }

  const sections = diff
    ? [
        `Precomputed git diff (read-only bridge output):\n${truncateReviewSection(
          diff,
          MAX_REVIEW_DIFF_CHARS,
          "[git diff truncated]",
        )}`,
      ]
    : [
        diffFailure === "outputLimit"
          ? "The git diff exceeded the bridge's 2,000,000-byte capture limit and was not included. Inspect the target directly with Read, Grep, and Glob."
          : diffFailure === "git"
            ? "No git diff was produced for this target because Git rejected the revision, path, or repository state. Inspect the target directly with Read, Grep, and Glob."
            : "No git diff was produced for this target. The repository may be clean; inspect the target directly with Read, Grep, and Glob.",
      ];

  if (untrackedFiles) {
    sections.push(
      `Worktree-wide untracked files may be unrelated to the requested target and are not included in the git diff; inspect only relevant files with Read:\n${truncateReviewSection(
        untrackedFiles,
        MAX_UNTRACKED_LIST_CHARS,
        "[untracked file list truncated]",
      )}`,
    );
  } else if (untrackedFailure === "outputLimit") {
    sections.push(
      "Worktree-wide untracked-file discovery exceeded the bridge's 2,000,000-byte capture limit; the list was not included. Use Glob to inspect potentially relevant files.",
    );
  } else if (untrackedFailure === "git") {
    sections.push(
      "Worktree-wide untracked-file discovery failed; use Glob to inspect potentially relevant files.",
    );
  }
  return sections.join("\n\n");
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export function createClaudeServer(
  runner: ClaudeRunner = runClaude,
  resolveWorkspaceIdentity: WorkspaceIdentityResolver = workspaceIdentity,
  resolveReviewContext: ReviewContextResolver = precomputedReviewContext,
): McpServer {
  const server = new McpServer({ name: "claude-bridge", version: "0.1.0" });
  const continuations = new Map<string, TaskContinuation>();

  const purgeExpiredContinuations = (now: number) => {
    for (const [token, continuation] of continuations) {
      if (!continuation.inFlight && continuation.expiresAt <= now) continuations.delete(token);
    }
  };

  const storeContinuation = (token: string, continuation: TaskContinuation) => {
    purgeExpiredContinuations(Date.now());
    if (!continuations.has(token) && continuations.size >= MAX_CONTINUATIONS) {
      const oldestIdleToken = [...continuations].find(([, entry]) => !entry.inFlight)?.[0];
      if (oldestIdleToken) continuations.delete(oldestIdleToken);
    }
    continuations.delete(token);
    continuations.set(token, continuation);
  };

  const runScopedTask = async (task: ScopedClaudeTask) => {
    const now = Date.now();
    purgeExpiredContinuations(now);

    let scope: string | undefined;
    let continuation: TaskContinuation | undefined;
    if (task.continuationToken) {
      scope = await taskScope(task, resolveWorkspaceIdentity);
      continuation = continuations.get(task.continuationToken);
      if (!continuation || continuation.scope !== scope) {
        return formatClaudeResponse(invalidContinuationResult());
      }
      if (continuation.inFlight) {
        return formatClaudeResponse(continuationInUseResult());
      }
      continuation.inFlight = true;
      continuation.expiresAt = now + CONTINUATION_TTL_MS;
    }

    let parsed: ClaudeResult;
    try {
      parsed = await runner(task.continuationToken ? CONTINUATION_PROMPT : task.prompt, {
        ...task.runOptions,
        workingDirectory: task.workingDirectory,
        maxTurns: task.continuationToken ? 50 : task.runOptions.maxTurns,
        resumeSessionId: continuation?.sessionId,
      });
    } catch (error) {
      if (continuation) continuation.inFlight = false;
      throw error;
    }

    if (parsed.subtype === "error_max_turns" && (parsed.sessionId || continuation?.sessionId)) {
      const token = task.continuationToken ?? randomUUID();
      const continuationScope = scope ?? (await taskScope(task, resolveWorkspaceIdentity));
      storeContinuation(token, {
        scope: continuationScope,
        sessionId: parsed.sessionId ?? continuation!.sessionId,
        expiresAt: Date.now() + CONTINUATION_TTL_MS,
        inFlight: false,
      });
      return formatClaudeResponse(parsed, token);
    }

    if (task.continuationToken && continuation) {
      const isTerminalSuccess =
        !parsed.isError &&
        !parsed.subtype?.startsWith("error_") &&
        (parsed.errors.length === 0 || parsed.resultText.length > 0);
      if (isTerminalSuccess) {
        continuations.delete(task.continuationToken);
        return formatClaudeResponse(parsed);
      }
      storeContinuation(task.continuationToken, {
        ...continuation,
        expiresAt: Date.now() + CONTINUATION_TTL_MS,
        inFlight: false,
      });
      return formatClaudeResponse(parsed, task.continuationToken);
    }
    return formatClaudeResponse(parsed);
  };

  server.registerTool(
    "claude_query",
    {
      title: "Ask Claude",
      description:
        "Ask Claude Code a question or give it a task. By default Claude can use its configured toolset; passing tools restricts built-in availability and disables configured MCP servers.",
      inputSchema: {
        prompt: z.string().describe("The question or task for Claude"),
        workingDirectory: z
          .string()
          .optional()
          .describe("Working directory (defaults to server cwd)"),
        model: z.enum(CLAUDE_MODELS).optional().describe("Claude model alias"),
        maxTurns: z
          .number()
          .int()
          .positive()
          .optional()
          .default(10)
          .describe("Maximum agentic turns (limits runtime)"),
        allowedTools: z
          .array(z.string())
          .optional()
          .describe(
            "Pre-approve permission prompts for specific tools; this does not restrict availability",
          ),
        tools: z
          .array(z.string())
          .nonempty()
          .optional()
          .describe(
            'Restrict available built-in tools via Claude CLI --tools (e.g., ["Read", "Grep"])',
          ),
        ...CONTINUATION_INPUT_SCHEMA,
      },
      outputSchema: CLAUDE_OUTPUT_SCHEMA,
    },
    async (
      {
        prompt,
        workingDirectory,
        model,
        maxTurns,
        maxBudgetUsd,
        tools,
        allowedTools,
        continuationToken,
      },
      extra,
    ) => {
      const progress = createProgressReporter(extra.sendNotification, extra._meta?.progressToken);
      return runScopedTask({
        toolName: "claude_query",
        taskIdentity: [prompt, model ?? null, tools ?? null, allowedTools ?? null],
        prompt,
        workingDirectory,
        continuationToken,
        runOptions: {
          model,
          maxTurns,
          maxBudgetUsd,
          disableMcpServers: tools !== undefined,
          tools,
          allowedTools,
          progress,
        },
      });
    },
  );

  server.registerTool(
    "claude_review_code",
    {
      title: "Claude Code Review",
      description:
        "Ask Claude to review code for quality, bugs, security issues, and best practices. Provide a git diff range, file paths, or code snippet.",
      inputSchema: {
        target: z.string().describe("What to review: git diff range, file paths, or code snippet"),
        focusAreas: z
          .string()
          .optional()
          .describe("Focus on: bugs, performance, style, security, etc."),
        context: z.string().optional().describe("Additional context about the changes"),
        workingDirectory: z.string().optional(),
        maxTurns: z.number().int().positive().optional().default(50),
        ...CONTINUATION_INPUT_SCHEMA,
      },
      outputSchema: CLAUDE_OUTPUT_SCHEMA,
    },
    async (
      { target, focusAreas, context, workingDirectory, maxTurns, maxBudgetUsd, continuationToken },
      extra,
    ) => {
      const progress = createProgressReporter(extra.sendNotification, extra._meta?.progressToken);
      let prompt = `Review the following code changes. Provide specific, actionable feedback with line references.\n\nTarget: ${target}`;
      if (focusAreas) prompt += `\n\nFocus areas: ${focusAreas}`;
      if (context) prompt += `\n\nContext: ${context}`;
      if (!continuationToken) {
        prompt += `\n\n${await resolveReviewContext(target, workingDirectory)}`;
      }

      return runScopedTask({
        toolName: "claude_review_code",
        taskIdentity: [target, focusAreas ?? null, context ?? null],
        prompt,
        workingDirectory,
        continuationToken,
        runOptions: {
          maxTurns,
          maxBudgetUsd,
          disableMcpServers: true,
          tools: READ_ONLY_AVAILABLE_TOOLS,
          allowedTools: READ_ONLY_ALLOWED_TOOLS,
          progress,
        },
      });
    },
  );

  server.registerTool(
    "claude_review_plan",
    {
      title: "Claude Plan Review",
      description:
        "Ask Claude to critique an implementation plan. Claude will examine the actual codebase to validate feasibility and consistency with existing patterns.",
      inputSchema: {
        plan: z.string().describe("The implementation plan to review"),
        codebasePath: z.string().optional().describe("Path to relevant codebase for context"),
        constraints: z.string().optional().describe("Known constraints"),
        workingDirectory: z.string().optional(),
        maxTurns: z.number().int().positive().optional().default(50),
        ...CONTINUATION_INPUT_SCHEMA,
      },
      outputSchema: CLAUDE_OUTPUT_SCHEMA,
    },
    async (
      {
        plan,
        codebasePath,
        constraints,
        workingDirectory,
        maxTurns,
        maxBudgetUsd,
        continuationToken,
      },
      extra,
    ) => {
      const progress = createProgressReporter(extra.sendNotification, extra._meta?.progressToken);
      let prompt = `Critique this implementation plan. Evaluate feasibility against the actual codebase, check consistency with existing patterns, identify gaps and risks.\n\nPlan:\n${plan}`;
      if (codebasePath) prompt += `\n\nRelevant codebase: ${codebasePath}`;
      if (constraints) prompt += `\n\nConstraints: ${constraints}`;

      return runScopedTask({
        toolName: "claude_review_plan",
        taskIdentity: [plan, codebasePath ?? null, constraints ?? null],
        prompt,
        workingDirectory,
        continuationToken,
        runOptions: {
          maxTurns,
          maxBudgetUsd,
          disableMcpServers: true,
          tools: READ_ONLY_AVAILABLE_TOOLS,
          allowedTools: READ_ONLY_ALLOWED_TOOLS,
          progress,
        },
      });
    },
  );

  server.registerTool(
    "claude_explain_code",
    {
      title: "Claude Explain Code",
      description:
        "Ask Claude to deeply explain code, logic, or architecture. Claude will read the actual source files to give grounded explanations.",
      inputSchema: {
        target: z
          .string()
          .describe("What to explain: file path, function name, module, or code snippet"),
        depth: z
          .enum(["overview", "detailed", "trace"])
          .optional()
          .default("detailed")
          .describe("Depth: overview, detailed, or full execution trace"),
        context: z.string().optional().describe("Additional context about the codebase"),
        workingDirectory: z.string().optional(),
        maxTurns: z.number().int().positive().optional().default(8),
        ...CONTINUATION_INPUT_SCHEMA,
      },
      outputSchema: CLAUDE_OUTPUT_SCHEMA,
    },
    async (
      { target, depth, context, workingDirectory, maxTurns, maxBudgetUsd, continuationToken },
      extra,
    ) => {
      const progress = createProgressReporter(extra.sendNotification, extra._meta?.progressToken);
      const prompt = buildExplainCodePrompt({ target, depth, context });
      return runScopedTask({
        toolName: "claude_explain_code",
        taskIdentity: [target, depth, context ?? null],
        prompt,
        workingDirectory,
        continuationToken,
        runOptions: {
          maxTurns,
          maxBudgetUsd,
          disableMcpServers: true,
          tools: READ_ONLY_AVAILABLE_TOOLS,
          allowedTools: READ_ONLY_ALLOWED_TOOLS,
          progress,
        },
      });
    },
  );

  server.registerTool(
    "claude_plan_perf",
    {
      title: "Claude Performance Plan",
      description:
        "Ask Claude to analyze performance and create an improvement plan. Claude reads the actual code to identify bottlenecks and propose optimizations.",
      inputSchema: {
        target: z.string().describe("What to optimize: function, module, or pipeline path"),
        metrics: z
          .array(z.enum(["latency", "throughput", "memory", "binary-size"]))
          .optional()
          .describe("Performance metrics to focus on"),
        constraints: z.string().optional().describe("Constraints"),
        context: z.string().optional().describe("Additional context about usage patterns"),
        workingDirectory: z.string().optional(),
        maxTurns: z.number().int().positive().optional().default(10),
        ...CONTINUATION_INPUT_SCHEMA,
      },
      outputSchema: CLAUDE_OUTPUT_SCHEMA,
    },
    async (
      {
        target,
        metrics,
        constraints,
        context,
        workingDirectory,
        maxTurns,
        maxBudgetUsd,
        continuationToken,
      },
      extra,
    ) => {
      const progress = createProgressReporter(extra.sendNotification, extra._meta?.progressToken);
      const prompt = buildPlanPerfPrompt({ target, metrics, constraints, context });
      return runScopedTask({
        toolName: "claude_plan_perf",
        taskIdentity: [target, metrics ?? null, constraints ?? null, context ?? null],
        prompt,
        workingDirectory,
        continuationToken,
        runOptions: {
          maxTurns,
          maxBudgetUsd,
          disableMcpServers: true,
          tools: READ_ONLY_AVAILABLE_TOOLS,
          allowedTools: READ_ONLY_ALLOWED_TOOLS,
          progress,
        },
      });
    },
  );

  server.registerTool(
    "claude_implement",
    {
      title: "Claude Implement",
      description:
        "Ask Claude to implement a feature, fix a bug, or make code changes. WARNING: This can modify files and run shell commands without per-command prompts.",
      inputSchema: {
        task: z.string().describe("What to implement or fix"),
        workingDirectory: z.string().optional(),
        model: z.enum(CLAUDE_MODELS).optional().describe("Claude model alias"),
        maxTurns: z.number().int().positive().optional().default(50),
        ...CONTINUATION_INPUT_SCHEMA,
      },
      outputSchema: CLAUDE_OUTPUT_SCHEMA,
    },
    async ({ task, workingDirectory, model, maxTurns, maxBudgetUsd, continuationToken }, extra) => {
      const progress = createProgressReporter(extra.sendNotification, extra._meta?.progressToken);
      return runScopedTask({
        toolName: "claude_implement",
        taskIdentity: [task, model ?? null],
        prompt: task,
        workingDirectory,
        continuationToken,
        runOptions: {
          model,
          maxTurns,
          maxBudgetUsd,
          disableMcpServers: true,
          tools: IMPLEMENTATION_AVAILABLE_TOOLS,
          allowedTools: IMPLEMENTATION_AVAILABLE_TOOLS,
          progress,
        },
      });
    },
  );

  return server;
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

export async function startClaudeServer(): Promise<void> {
  const server = createClaudeServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info("claude-bridge MCP server started on stdio");
}

const entryPoint = process.argv[1];
if (entryPoint && realpathSync(entryPoint) === realpathSync(fileURLToPath(import.meta.url))) {
  startClaudeServer().catch((err) => {
    logger.error("Failed to start claude-bridge:", err);
    process.exit(1);
  });
}
