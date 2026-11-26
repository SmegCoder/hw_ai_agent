---
name: open-task-export
description: Export open tasks from a study tracker to JSON. Use for open or unfinished task lists and do not use for overdue-only reports.
---

# Export open tasks

1. Use the source named by the user. For the study tracker, call its MCP `list_tasks` tool when available. If no tracker or source is specified, look for `tasks.json` at the workspace root. Accept a JSON array or an object with a `tasks` array. If the selected source fails, report the error and ask for a usable source. An empty array is a valid source.
2. Keep tasks whose trimmed, case-insensitive status is `open`, `todo`, `in_progress` or `blocked`. Omit `done`, `closed`, `completed` and `cancelled`. A task must have an `id` and a non-empty `title`. Skip records without those fields, with missing or unknown statuses, or that are not objects, and report how many were skipped. Do not infer statuses from titles or dates.
3. Preserve source order and the original `id` and `title`. Export each item as an object with `id`, `title`, normalized `status` and `due_date`. Copy a valid ISO date (`YYYY-MM-DD`) as `due_date`, use `null` when it is absent or invalid. Do not fill missing dates by guessing.
4. Follow the requested output shape. For a unique ID list, keep the first occurrence of each qualifying ID. By default write `{"tasks": [...]}` to `open-tasks.json`. Keep the source intact and follow the usual write approval. Report the number of exported tasks and any skipped records. For an inline request, return the list without writing a file.
