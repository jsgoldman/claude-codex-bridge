---
name: claude
description: Use when the user asks Claude Code to review, explain, plan, analyze performance, answer a question, or explicitly implement or fix code.
---

Route the request to the most specific Claude MCP tool. Claude can perform read-only analysis or, through the explicit implementation tool, edit and test code.

## Tool Selection

Pick the best tool based on the user's request:

| Request Type                | Tool                               | Key Parameters                                   |
| --------------------------- | ---------------------------------- | ------------------------------------------------ |
| Code review, diff review    | `mcp__claude__claude_review_code`  | structured `target`, `focusAreas`                |
| Plan critique               | `mcp__claude__claude_review_plan`  | `plan`, repository-relative `codebasePath`       |
| Explain code                | `mcp__claude__claude_explain_code` | `target`, `paths`, `depth`                       |
| Performance analysis        | `mcp__claude__claude_plan_perf`    | `target`, `paths`, `metrics`                     |
| Implement/fix (writes code) | `mcp__claude__claude_implement`    | `task`                                           |
| General question            | `mcp__claude__claude_query`        | `prompt`, `paths` when nested instructions apply |

## Instructions

1. Parse the user's request to determine the task type
2. If the user references files, read them first for context
3. Call the most specific Claude tool — prefer specialized tools over `claude_query`
4. Always pass `workingDirectory` to every tool call
5. Synthesize the response: summarize key findings, highlight important points, give actionable recommendations
6. Only use `claude_implement` if the user explicitly asks Claude to make changes

## Sessions and results

- Start every independent task fresh. Do not supply a continuation token from another invocation.
- Every result exposes `session_id`, `num_turns`, `subtype`, `is_error`, `cost`, and `errors`; report them when diagnosing incomplete or expensive runs.
- If a result includes `continuation_token`, call the same tool again with that value passed as `continuationToken`, identical task-defining arguments, and the same `workingDirectory`. The bridge resumes that exact Claude session with 50 turns. A transient resumed-call failure returns the same token so paid work is not lost.
- Never reuse a continuation token for another task, tool, branch, PR, or worktree. The bridge rejects a mismatched scope.
- Use optional `maxBudgetUsd` when the user wants a hard spend ceiling. Tune future `maxTurns` values from observed `num_turns`; review and implementation tools default to 50.

The bridge runs `claude_query` and all review/analysis tools in Claude safe mode with only `Read`, `Grep`, and `Glob`, auto-memory disabled, no hooks/plugins/settings or slash commands, a fail-closed headless permission mode, and an empty strict MCP config. Safe mode preserves subscription authentication. The bridge supplies a capped JSON bundle of repository-owned root and scoped `AGENTS.md`/`CLAUDE.md` instructions through stdin; tracked in-tree symlinks are expanded and deduplicated. `.claude/CLAUDE.md` is intentionally excluded because `.claude` is a customization source. General analysis uses committed HEAD and the explicit `paths`/`codebasePath` scopes, while code review uses its structured target and the trusted base revision for a Git range. The bundle revision and digest are bound to continuation scope. Incomplete, cyclic, or escaping instructions, an invalid trusted revision, a truncated or capture-limited diff, or an incomplete untracked-file inventory returns a structured error without invoking Claude. `claude_implement` alone retains and pre-approves edit, write, and command tools so it can build and verify code, and the bridge never automatically replays a failed implementation subprocess.

Code review targets are a discriminated union; never infer a target kind from an arbitrary string:

- Git range: `{ "kind": "gitRange", "base": "HEAD~1", "head": "HEAD" }`
- Working tree or files: `{ "kind": "paths", "paths": ["."] }` or explicit repository-relative paths
- Symbol: `{ "kind": "symbol", "symbol": "parseResult" }`
- Snippet: `{ "kind": "snippet", "code": "const value = 1", "language": "typescript" }`

## Examples

- `/claude review my recent changes` → `claude_review_code` with target `{ "kind": "gitRange", "base": "HEAD~1", "head": "HEAD" }`
- `/claude explain src/lib/exec.ts` → `claude_explain_code` with target "src/lib/exec.ts" and paths `["src/lib/exec.ts"]`
- `/claude is my approach to caching correct?` → `claude_query` with the question and any relevant repository-relative `paths`
- `/claude optimize the response parsing` → `claude_plan_perf` with target plus relevant `paths`
- `/claude implement error handling for timeouts` → `claude_implement` with task
