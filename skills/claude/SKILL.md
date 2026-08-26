---
name: claude
description: Use when the user asks Claude Code to review, explain, plan, analyze performance, answer a question, or explicitly implement or fix code.
---

Route the request to the most specific Claude MCP tool. Claude can perform read-only analysis or, through the explicit implementation tool, edit and test code.

## Tool Selection

Pick the best tool based on the user's request:

| Request Type                | Tool                               | Key Parameters                              |
| --------------------------- | ---------------------------------- | ------------------------------------------- |
| Code review, diff review    | `mcp__claude__claude_review_code`  | `target` (diff range or file), `focusAreas` |
| Plan critique               | `mcp__claude__claude_review_plan`  | `plan`, `codebasePath`                      |
| Explain code                | `mcp__claude__claude_explain_code` | `target` (file/function), `depth`           |
| Performance analysis        | `mcp__claude__claude_plan_perf`    | `target`, `metrics`                         |
| Implement/fix (writes code) | `mcp__claude__claude_implement`    | `task`                                      |
| General question            | `mcp__claude__claude_query`        | `prompt`                                    |

## Instructions

1. Parse the user's request to determine the task type
2. If the user references files, read them first for context
3. Call the most specific Claude tool — prefer specialized tools over `claude_query`
4. Always pass `workingDirectory` to every tool call
5. Synthesize the response: summarize key findings, highlight important points, give actionable recommendations
6. Only use `claude_implement` if the user explicitly asks Claude to make changes

## Sessions and results

- Start every independent task fresh. Do not supply a continuation token from another invocation.
- Every result exposes `session_id`, `num_turns`, `subtype`, `is_error`, and `cost`; report them when diagnosing incomplete or expensive runs.
- If a result includes `continuation_token`, call the same tool again with that value passed as `continuationToken`, identical task-defining arguments, and the same `workingDirectory`. The bridge resumes that exact Claude session with 50 turns. A transient resumed-call failure returns the same token so paid work is not lost.
- Never reuse a continuation token for another task, tool, branch, PR, or worktree. The bridge rejects a mismatched scope.
- Use optional `maxBudgetUsd` when the user wants a hard spend ceiling. Tune future `maxTurns` values from observed `num_turns`; review and implementation tools default to 50.

The bridge restricts actual review/analysis tool availability to `Read`, `Grep`, and `Glob` with Claude CLI `--tools` and an empty strict MCP config, so unrelated configured MCP tools are unavailable. Code-review diffs are computed by the bridge and included in the prompt; unavailable, empty, capture-limited, truncated, and worktree-wide untracked-file cases are explicitly labeled for direct inspection. `allowedTools` only pre-approves permission prompts. `claude_implement` retains and pre-approves explicit read, edit, write, and command tools so it can build and verify code.

## Examples

- `/claude review my recent changes` → `claude_review_code` with target "HEAD~1..HEAD"
- `/claude explain src/lib/exec.ts` → `claude_explain_code` with target "src/lib/exec.ts"
- `/claude is my approach to caching correct?` → `claude_query` with the question
- `/claude optimize the response parsing` → `claude_plan_perf` with target
- `/claude implement error handling for timeouts` → `claude_implement` with task
