import { open, readFile, realpath, rename, rm, stat } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const SCHEMA_VERSION = 1;
const READER_POLICY_ID = "hyparquet_swebench_metadata_projection_v1";
const COLUMNS = ["instance_id", "repo", "problem_statement", "difficulty"];

function fail(message) {
  throw new Error(message);
}

function inside(root, candidate, label) {
  const value = relative(root, candidate);
  if (!value || value === ".." || value.startsWith(`..\\`) || isAbsolute(value)) {
    fail(`${label} must be a file below the repository root`);
  }
}

async function main() {
  const [inputArgument, outputArgument, moduleArgument] = process.argv.slice(2);
  if (!inputArgument || !outputArgument || !moduleArgument) {
    fail("usage: node extract_swebench_metadata.mjs INPUT OUTPUT HYPARQUET_NODE_JS");
  }
  const root = await realpath(process.cwd());
  const input = await realpath(resolve(root, inputArgument));
  const modulePath = await realpath(resolve(root, moduleArgument));
  const output = resolve(root, outputArgument);
  inside(root, input, "input");
  inside(root, modulePath, "hyparquet module");
  inside(root, output, "output");
  if (dirname(output) === root) fail("output must use a dedicated workspace directory");
  try {
    await stat(output);
    fail("output already exists");
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }

  const packagePath = resolve(dirname(modulePath), "..", "package.json");
  const packageDocument = JSON.parse(await readFile(packagePath, "utf8"));
  if (packageDocument.name !== "hyparquet" || packageDocument.version !== "1.26.2") {
    fail("hyparquet must be exactly version 1.26.2");
  }
  const { asyncBufferFromFile, parquetReadObjects } = await import(
    pathToFileURL(modulePath).href
  );
  const file = await asyncBufferFromFile(input);
  const rows = await parquetReadObjects({ file, columns: COLUMNS });
  if (!Array.isArray(rows) || rows.length === 0) fail("parquet metadata is empty");
  const tasks = rows.map((row) => {
    const instanceId = String(row.instance_id ?? "").trim();
    const repo = String(row.repo ?? "").trim();
    const problemStatement = String(row.problem_statement ?? "");
    const difficulty = row.difficulty == null ? null : String(row.difficulty).trim();
    if (!instanceId || !repo || !problemStatement) fail("parquet task metadata is incomplete");
    return {
      instance_id: instanceId,
      repo,
      problem_statement: problemStatement,
      difficulty: difficulty || null,
    };
  });
  tasks.sort((left, right) => left.instance_id.localeCompare(right.instance_id));
  if (new Set(tasks.map((task) => task.instance_id)).size !== tasks.length) {
    fail("parquet task metadata contains duplicate instance ids");
  }
  const inputStat = await stat(input);
  const document = {
    schema_version: SCHEMA_VERSION,
    reader: {
      policy_id: READER_POLICY_ID,
      package: "hyparquet",
      version: packageDocument.version,
      columns: COLUMNS,
    },
    source: {
      bytes: inputStat.size,
    },
    task_count: tasks.length,
    tasks,
  };
  const temporary = `${output}.tmp-${process.pid}`;
  let handle;
  try {
    handle = await open(temporary, "wx");
    await handle.writeFile(`${JSON.stringify(document)}\n`, "utf8");
    await handle.sync();
    await handle.close();
    handle = undefined;
    await rename(temporary, output);
  } finally {
    if (handle) await handle.close();
    await rm(temporary, { force: true });
  }
}

await main();
