import { spawn } from "node:child_process";
import { unlink } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { watch } from "chokidar";

const websiteDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const rootDir = path.dirname(websiteDir);
const generatedData = path.join(websiteDir, "src/generated/scaling-laws.json");
const watchedPaths = [
  path.join(rootDir, "experiments"),
  path.join(rootDir, "configs"),
  path.join(rootDir, "src/chess_engine_4/model"),
  path.join(rootDir, "src/chess_engine_4/training/generate_website_data.py"),
  path.join(rootDir, "src/chess_engine_4/training/flops.py"),
  path.join(rootDir, "src/chess_engine_4/training/scaling_laws.py"),
];

const next = spawn("pnpm", ["exec", "next", "dev", ...process.argv.slice(2)], {
  cwd: websiteDir,
  stdio: "inherit",
});
const watcher = watch(watchedPaths, { ignoreInitial: true });

let exporter;
let exportPending = false;
let debounceTimer;

watcher.on("all", (_event, changedPath) => {
  if (!isScalingInput(changedPath)) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(
    () => regenerateScalingData(path.relative(rootDir, changedPath)),
    100,
  );
});

function isScalingInput(changedPath) {
  const relativePath = path.relative(rootDir, changedPath);
  return (
    /^experiments\/best-runs-[^/]+\.toml$/.test(relativePath) ||
    /^experiments\/throughput-[^/]+\.toml$/.test(relativePath) ||
    /^configs\/[^/]+\.py$/.test(relativePath) ||
    /^src\/chess_engine_4\/model\/.*\.py$/.test(relativePath) ||
    relativePath === "src/chess_engine_4/training/generate_website_data.py" ||
    relativePath === "src/chess_engine_4/training/flops.py" ||
    relativePath === "src/chess_engine_4/training/scaling_laws.py"
  );
}

function regenerateScalingData(changedPath) {
  if (exporter) {
    exportPending = true;
    return;
  }

  console.log(`[scaling-data] regenerating after ${changedPath}`);
  exporter = spawn("uv", ["run", "generate-website-data"], {
    cwd: rootDir,
    stdio: "inherit",
  });
  exporter.on("exit", async (code) => {
    if (code !== 0) {
      try {
        await unlink(generatedData);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
      console.error(
        `[scaling-data] generation failed with code ${code}; removed stale generated data`,
      );
    }
    exporter = undefined;
    if (exportPending) {
      exportPending = false;
      regenerateScalingData("additional changes");
    }
  });
}

async function stop(signal) {
  clearTimeout(debounceTimer);
  await watcher.close();
  exporter?.kill(signal);
  next.kill(signal);
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => void stop(signal));
}

next.on("exit", async (code) => {
  await watcher.close();
  exporter?.kill("SIGTERM");
  process.exit(code ?? 1);
});
