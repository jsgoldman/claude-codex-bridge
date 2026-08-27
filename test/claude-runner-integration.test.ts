import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { runClaude } from "../src/claude-server.js";

describe("runClaude process boundary", () => {
  const originalPath = process.env["PATH"];

  afterEach(() => {
    process.env["PATH"] = originalPath;
    delete process.env["FAKE_CLAUDE_ARGS"];
    delete process.env["FAKE_CLAUDE_ENV"];
    delete process.env["FAKE_CLAUDE_STDIN"];
  });

  it("passes safe read-only argv, disabled auto-memory, and the instruction prompt through stdin", async () => {
    const tempDirectory = mkdtempSync(join(tmpdir(), "ccb-fake-claude-"));
    const executable = join(tempDirectory, "claude");
    const argsPath = join(tempDirectory, "args.txt");
    const envPath = join(tempDirectory, "env.txt");
    const stdinPath = join(tempDirectory, "stdin.txt");
    writeFileSync(
      executable,
      '#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_CLAUDE_ARGS"\nprintf "%s" "$CLAUDE_CODE_DISABLE_AUTO_MEMORY" > "$FAKE_CLAUDE_ENV"\ncat > "$FAKE_CLAUDE_STDIN"\nprintf "{\\"result\\":\\"ok\\",\\"is_error\\":false}"\n',
    );
    chmodSync(executable, 0o755);
    process.env["PATH"] = `${tempDirectory}:${originalPath}`;
    process.env["FAKE_CLAUDE_ARGS"] = argsPath;
    process.env["FAKE_CLAUDE_ENV"] = envPath;
    process.env["FAKE_CLAUDE_STDIN"] = stdinPath;

    try {
      const result = await runClaude("ROOT_POLICY\n\nInspect", {
        safeMode: true,
        disableAutoMemory: true,
        disableSlashCommands: true,
        permissionMode: "dontAsk",
        disableMcpServers: true,
        tools: ["Read", "Grep", "Glob"],
        maxRetries: 0,
      });

      expect(result.isError).toBe(false);
      expect(readFileSync(argsPath, "utf8").trim().split("\n")).toEqual([
        "-p",
        "--output-format",
        "json",
        "--safe-mode",
        "--disable-slash-commands",
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--tools",
        "Read,Grep,Glob",
      ]);
      expect(readFileSync(envPath, "utf8")).toBe("1");
      expect(readFileSync(stdinPath, "utf8")).toBe("ROOT_POLICY\n\nInspect");
    } finally {
      rmSync(tempDirectory, { recursive: true, force: true });
    }
  });
});
