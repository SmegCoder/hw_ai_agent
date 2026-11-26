---
name: deadline-report
description: Make a report of unfinished study tasks overdue as of a specific date, including cases with missing or invalid due dates.
---

# Report overdue tasks

1. Use the user's reference date and normalize it to `YYYY-MM-DD`. If the date is missing or ambiguous, ask for it before writing a report. For the study tracker, call its MCP `list_tasks` tool when available. For a file request, read the indicated file or `tasks.json` at the workspace root. Accept a JSON array or an object with a `tasks` array. If the selected source fails, report the error and ask for a usable source.
2. Treat only `open`, `todo`, `in_progress` and `blocked` (trimmed and case-insensitive) as unfinished. A valid item needs an `id` and a non-empty `title`. Ignore completed and cancelled tasks. Count records with unknown status or missing identity as skipped. For unfinished tasks with an absent or invalid `due_date`, count a missing deadline and leave them outside the overdue list.
3. A task is overdue when its valid calendar `due_date` is strictly earlier than the reference date. Today's deadlines and future deadlines are excluded. Sort the results by `due_date` and then by the string form of `id`. Include `id`, `title`, `due_date` and `days_overdue`, the number of calendar days between the two dates.
4. Follow the requested output shape. If the user asks for missing-deadline IDs, report them separately from overdue items. By default write an object with `as_of`, `overdue` (array), `missing_due_date` (count) and `skipped_records` (count) to `deadlines.json`. Keep the source intact and follow the normal approval for writing. For an inline request, return the report without writing. Briefly report how many tasks are overdue and how many unfinished tasks have an unknown deadline.
