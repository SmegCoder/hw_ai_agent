---
name: grammar-check
description: Check grammar, spelling and punctuation in a text file and report the issues, optionally applying corrections, including cases of inconsistent terms and formatting.
---

# Check grammar in files

1. Use the user's target file and mode: report only, or report plus edits. If the file or the mode is missing or ambiguous, ask before editing. Read the indicated file or the text file named at the workspace root. Accept plain text such as `.md`, `.txt` and `.rst`. If the selected source fails, report the error and ask for a usable source.
2. Treat only genuine text problems as issues: spelling, grammar (agreement, tense and aspect), punctuation, clarity and consistency of terms and formats. Ignore code blocks, inline code, URLs and frontmatter. Count unreadable, binary or ambiguous fragments as skipped. For wording that is correct but debatable, list it separately as optional and keep it outside the issue list.
3. Order the issues by line number and then by position in the line. Include `line`, `current`, `suggested`, `category` and a short `reason`. For a whole-file pattern such as an inconsistent term, use the first occurrence as the location and note the rest.
4. Follow the requested output shape. If the user asks for issues of one category only, report them separately from the rest. By default write an object with `source`, `issues` (array), `optional` (array) and `skipped_fragments` (count) to `grammar-report.json`. Apply edits only when asked, and only to spelling, grammar and punctuation; do not change meaning, structure or numbers. Keep the source intact and follow the normal approval for writing. For an inline request, return the report without writing. Briefly report how many issues were found and how many parts were skipped.
