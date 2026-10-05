export const ALLOWED_TOOL_NAMES = Object.freeze([
  "edit",
  "glob",
  "grep",
  "job_kill",
  "job_list",
  "job_output",
  "pwsh",
  "read",
  "skill",
  "str_replace_editor",
  "todo_write",
  "write",
])

const allowedToolNames = new Set(ALLOWED_TOOL_NAMES)
const groupedPromptSections = new Map([
  ["jobs", ["job_kill", "job_list", "job_output"]],
])

export const TOOL_PROMPT_SECTION = Object.freeze({
  name: "dsh:tool-surface",
  text: "Use only the tool schemas included in the current Request. Tool availability may change after a router decision. When the task explicitly requests an available tool call, perform it before giving the terminal response. Delegation, workflow, goal-control, interaction, web, and image tools are unavailable in this DSH actor.",
})

export const WORKSPACE_PROMPT_SECTION = Object.freeze({
  name: "dsh:workspace-root",
  text: "The current working directory is the root of the repository checkout, and tool paths resolve against it. Pass paths relative to that root, without the repository name in front of them. When a path is not found, locate the file with glob and a pattern such as `**/<file name>`.",
})

export const WORKSPACE_PROMPT_SECTION_V2 = Object.freeze({
  name: "dsh:workspace-root-v2",
  text: "DSH tool guidance policy: swebench-source-patch-submit-v5. The current working directory is the repository checkout root; tool paths are relative to it. There is no `ls` tool. If the task asks you to run `ls`, use `pwsh` with `Get-ChildItem -Name`, or use `glob` to list files. After any `read`, `grep`, or `glob` says a path is missing, do not retry that path or guess a nearby one. Search from the checkout root with `glob` using `path: \".\"` and a pattern such as `**/<file-name>`; then use an actual returned path. Do not prefix a path with the repository name.",
})

export const WORKSPACE_PROMPT_SECTION_V3 = Object.freeze({
  name: "dsh:workspace-root-v3",
  text: "DSH tool guidance policy: swebench-source-patch-submit-v6. The `pwsh` tool is available in the first Request; `glob` and `grep` may appear only after a completed tool call. There is no `ls` tool. The current working directory is the repository checkout root. Before the first source `read`, locate the file with `pwsh` using `command: \"rg --files -g '*<filename>*'\"` and `description: \"Locate matching source file paths\"`; replace `<filename>` with the filename you seek. To list the checkout root, use `pwsh` with `command: \"Get-ChildItem -Name\"` and a short `description`. Use a returned relative path verbatim as `read.file_path`; do not prepend the repository folder name or repeat any path component. If `read` reports a missing path, locate matching filenames with `pwsh` before the next `read`. Never retry or guess a path that already failed. Use only tools included in the current Request.",
})

const WORKSPACE_PROMPT_PROVIDERS = new Set(["local-qwen"])

export function filterTools(tools = []) {
  return tools.filter((tool) => allowedToolNames.has(tool.name))
}

export function applyToolPolicy(assembly, provider, systemPromptPolicyId) {
  const tools = filterTools(assembly.tools)
  const requestToolNames = new Set(tools.map((tool) => tool.name))
  const sectionMatchesRequest = (section) => {
    if (!section.name.startsWith("tool:")) return true
    const sectionName = section.name.slice("tool:".length)
    if (requestToolNames.has(sectionName)) return true
    return groupedPromptSections.get(sectionName)?.some((name) => requestToolNames.has(name))
      ?? false
  }
  return {
    ...assembly,
    sections: [
      ...assembly.sections.filter(sectionMatchesRequest),
      ...(WORKSPACE_PROMPT_PROVIDERS.has(provider) ? [{ ...(
        systemPromptPolicyId === "swebench-source-patch-submit-v6"
          ? WORKSPACE_PROMPT_SECTION_V3
          : systemPromptPolicyId === "swebench-source-patch-submit-v5"
            ? WORKSPACE_PROMPT_SECTION_V2 : WORKSPACE_PROMPT_SECTION
      ) }] : []),
      { ...TOOL_PROMPT_SECTION },
    ],
    tools,
  }
}
