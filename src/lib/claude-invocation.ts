export interface ClaudeInvocationOptions {
  sessionId?: string;
  model?: string;
  maxTurns?: number;
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
  if (options.sessionId) args.push("--resume", options.sessionId);
  if (options.model) args.push("--model", options.model);
  if (options.maxTurns) args.push("--max-turns", String(options.maxTurns));
  if (options.allowedTools && options.allowedTools.length > 0) {
    for (const tool of options.allowedTools) {
      args.push("--allowedTools", tool);
    }
  }
  return { args, stdin: prompt };
}
