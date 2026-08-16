import { logger } from "./logger.js";
import type { ClaudeResult } from "./types.js";

/**
 * Parses the JSON output from `claude -p --output-format json`.
 * Claude Code versions may emit either one result object or an event array
 * whose terminal `{ type: "result" }` object contains the response content.
 */
export function parseClaudeOutput(jsonOutput: string): ClaudeResult {
  const result: ClaudeResult = {
    resultText: "",
    sessionId: null,
    costUsd: null,
    errors: [],
  };

  const trimmed = jsonOutput.trim();
  if (!trimmed) {
    result.errors.push("Empty output from Claude CLI");
    return result;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed) as unknown;
  } catch {
    // Claude may have output plain text instead of JSON
    logger.debug("Failed to parse Claude output as JSON, using raw text");
    result.resultText = trimmed;
    return result;
  }

  const payload = terminalResultPayload(parsed);
  if (payload === undefined) {
    result.resultText = JSON.stringify(parsed, null, 2);
    return result;
  }

  // Extract result text from structured output
  // Claude JSON format: { result: string, ... } or { result: { content: [...] }, ... }
  const resultField = payload["result"];
  if (typeof resultField === "string") {
    result.resultText = resultField;
  } else if (resultField && typeof resultField === "object") {
    const content = (resultField as Record<string, unknown>)["content"] as
      | Array<Record<string, unknown>>
      | undefined;
    if (Array.isArray(content)) {
      result.resultText = content
        .filter((c) => c["type"] === "text")
        .map((c) => c["text"] as string)
        .join("\n");
    }
  }

  // If result field didn't yield text, try other common fields
  if (!result.resultText) {
    const message = payload["message"] as string | undefined;
    const text = payload["text"] as string | undefined;
    const output = payload["output"] as string | undefined;
    result.resultText = message ?? text ?? output ?? "";
  }

  // If still nothing, stringify the whole response
  if (!result.resultText && payload["is_error"] !== true && Object.keys(payload).length > 0) {
    result.resultText = JSON.stringify(payload, null, 2);
  }

  // Extract metadata
  result.sessionId = (payload["session_id"] as string) ?? (payload["sessionId"] as string) ?? null;
  result.costUsd =
    (payload["total_cost_usd"] as number) ??
    (payload["cost_usd"] as number) ??
    (payload["costUsd"] as number) ??
    null;

  // Check for errors
  const error = payload["error"] as string | Record<string, unknown> | undefined;
  if (error) {
    const msg =
      typeof error === "string" ? error : ((error["message"] as string) ?? JSON.stringify(error));
    result.errors.push(msg);
  }

  const errors = payload["errors"];
  if (Array.isArray(errors)) {
    result.errors.push(...errors.filter((error): error is string => typeof error === "string"));
  }

  return result;
}

function terminalResultPayload(parsed: unknown): Record<string, unknown> | undefined {
  if (isRecord(parsed)) return parsed;
  if (!Array.isArray(parsed)) return undefined;

  for (let index = parsed.length - 1; index >= 0; index -= 1) {
    const event = parsed[index];
    if (isRecord(event) && event["type"] === "result") return event;
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
