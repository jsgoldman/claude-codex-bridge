export interface ClaudeInvocationOptions {
  resumeSessionId?: string;
  model?: string;
  maxTurns?: number;
  maxBudgetUsd?: number;
  disableMcpServers?: boolean;
  tools?: string[];
  allowedTools?: string[];
}

export interface ClaudeInvocation {
  args: string[];
  stdin: string | undefined;
}

/** Build one non-interactive Claude CLI invocation. */
export function buildClaudeInvocation(
  prompt: string,
  options: ClaudeInvocationOptions = {},
): ClaudeInvocation {
  const args = ["-p", "--output-format", "json"];
  if (options.resumeSessionId) args.push("--resume", options.resumeSessionId);
  if (options.model) args.push("--model", options.model);
  if (options.maxTurns !== undefined) args.push("--max-turns", String(options.maxTurns));
  if (options.maxBudgetUsd !== undefined) {
    args.push("--max-budget-usd", String(options.maxBudgetUsd));
  }
  if (options.disableMcpServers) {
    args.push("--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}');
  }
  if (options.tools !== undefined) args.push("--tools", options.tools.join(","));
  if (options.allowedTools && options.allowedTools.length > 0) {
    for (const tool of options.allowedTools) {
      args.push("--allowedTools", tool);
    }
  }
  return { args, stdin: prompt };
}
