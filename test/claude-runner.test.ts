import { beforeEach, describe, expect, it, vi } from "vitest";

const execCommand = vi.hoisted(() => vi.fn());

vi.mock("../src/lib/exec-runner.js", () => ({ execCommand }));

import { runClaude } from "../src/claude-server.js";

describe("runClaude", () => {
  beforeEach(() => {
    execCommand.mockReset();
  });

  it("marks a nonzero process exit as an error even when stdout is nonempty and malformed", async () => {
    execCommand.mockResolvedValue({
      exitCode: 1,
      stdout: "malformed diagnostic stdout",
      stderr: "fatal transport diagnostic",
      timedOut: false,
    });

    const result = await runClaude("Inspect the repository");

    expect(result).toMatchObject({
      resultText: "malformed diagnostic stdout",
      isError: true,
      errors: ["fatal transport diagnostic"],
    });
  });
});
