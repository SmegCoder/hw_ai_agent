# Evals

Модель openrouter/deepseek/deepseek-v4.1-flash. Сохранено попыток 30 из 30.

Скиллы повысят долю правильных результатов в сценариях отбора задач и дедлайнов. Дополнительные инструкции могут увеличить расход входных токенов.

## Результаты

| Сценарий          | A/B | Результат | Вызовы | Ошибки выбора | Инстр. | MCP | Сбои |
| ----------------- | --- | --------: | -----: | ------------: | -----: | --: | ---: |
| open-tasks        | A   |       3/3 |    3/3 |             — |      6 |   3 |    0 |
| open-tasks        | B   |       3/3 |    3/3 |             0 |      9 |   3 |    0 |
| deadlines         | A   |       2/3 |    2/3 |             — |     11 |   3 |    1 |
| deadlines         | B   |       3/3 |    3/3 |             0 |     10 |   3 |    0 |
| note              | A   |       3/3 |    2/3 |             — |      7 |   0 |    1 |
| note              | B   |       3/3 |    0/3 |             0 |      6 |   0 |    3 |
| missing-deadlines | A   |       3/3 |    3/3 |             — |      8 |   3 |    0 |
| missing-deadlines | B   |       3/3 |    3/3 |             0 |      9 |   3 |    0 |
| mcp-failure       | A   |       3/3 |    3/3 |             — |      3 |   3 |    3 |
| mcp-failure       | B   |       3/3 |    3/3 |             0 |      6 |   3 |    3 |
| Итого             | A   |     14/15 |  13/15 |             — |     35 |  12 |    5 |
| Итого             | B   |     15/15 |  12/15 |             0 |     40 |  12 |    6 |

В столбце «Вызовы» указано число попыток, прошедших проверку инструментов. В столбце «Сбои» учтены и ожидаемые ошибки MCP.

## Токены API

| Сценарий          | A/B |  Вход | Ср. вход | API вход | Выход | Ср. выход | API выход |
| ----------------- | --- | ----: | -------: | -------: | ----: | --------: | --------: |
| open-tasks        | A   | 14039 |   4679.7 |      9/9 |   744 |     248.0 |       9/9 |
| open-tasks        | B   | 18442 |   6147.3 |      9/9 |  1000 |     333.3 |       9/9 |
| deadlines         | A   | 17230 |   5743.3 |    10/10 |  2081 |     693.7 |     10/10 |
| deadlines         | B   | 19670 |   6556.7 |      9/9 |  1586 |     528.7 |       9/9 |
| note              | A   | 11432 |   3810.7 |      9/9 |  1022 |     340.7 |       9/9 |
| note              | B   | 13312 |   4437.3 |      9/9 |  1021 |     340.3 |       9/9 |
| missing-deadlines | A   | 16373 |   5457.7 |      9/9 |  1438 |     479.3 |       9/9 |
| missing-deadlines | B   | 22662 |   7554.0 |    10/10 |  1554 |     518.0 |     10/10 |
| mcp-failure       | A   |  6919 |   2306.3 |      6/6 |   292 |      97.3 |       6/6 |
| mcp-failure       | B   |  9383 |   3127.7 |      6/6 |   403 |     134.3 |       6/6 |
| Итого             | A   | 65993 |   4399.5 |    43/43 |  5577 |     371.8 |     43/43 |
| Итого             | B   | 83469 |   5564.6 |    43/43 |  5564 |     370.9 |     43/43 |

Колонки API показывают долю обращений, для которых провайдер вернул usage. Если usage известен частично, в сумме только известные значения. Средние рассчитаны по попыткам с полным usage.

## Попытки

Номер попытки открывает журнал.

| Сценарий          | Попытка       | Результат | Вызовы | Выбор | Скиллы               | MCP | Сбой |
| ----------------- | ------------- | --------- | ------ | ----- | -------------------- | --: | ---- |
| open-tasks        | [A1][run-001] | да        | да     | —     | 0                    |   1 | —    |
| open-tasks        | [B1][run-002] | да        | да     | да    | open-task-export × 1 |   1 | —    |
| open-tasks        | [A2][run-003] | да        | да     | —     | 0                    |   1 | —    |
| open-tasks        | [B2][run-004] | да        | да     | да    | open-task-export × 1 |   1 | —    |
| open-tasks        | [A3][run-005] | да        | да     | —     | 0                    |   1 | —    |
| open-tasks        | [B3][run-006] | да        | да     | да    | open-task-export × 1 |   1 | —    |
| deadlines         | [A1][run-007] | да        | да     | —     | 0                    |   1 | —    |
| deadlines         | [B1][run-008] | да        | да     | да    | deadline-report × 1  |   1 | —    |
| deadlines         | [A2][run-009] | да        | да     | —     | 0                    |   1 | —    |
| deadlines         | [B2][run-010] | да        | да     | да    | deadline-report × 1  |   1 | —    |
| deadlines         | [A3][run-011] | нет       | нет    | —     | 0                    |   1 | E1   |
| deadlines         | [B3][run-012] | да        | да     | да    | deadline-report × 1  |   1 | —    |
| note              | [A1][run-013] | да        | да     | —     | 0                    |   0 | —    |
| note              | [B1][run-014] | да        | нет    | да    | 0                    |   0 | E2   |
| note              | [A2][run-015] | да        | да     | —     | 0                    |   0 | —    |
| note              | [B2][run-016] | да        | нет    | да    | 0                    |   0 | E2   |
| note              | [A3][run-017] | да        | нет    | —     | 0                    |   0 | E2   |
| note              | [B3][run-018] | да        | нет    | да    | 0                    |   0 | E2   |
| missing-deadlines | [A1][run-019] | да        | да     | —     | 0                    |   1 | —    |
| missing-deadlines | [B1][run-020] | да        | да     | да    | deadline-report × 1  |   1 | —    |
| missing-deadlines | [A2][run-021] | да        | да     | —     | 0                    |   1 | —    |
| missing-deadlines | [B2][run-022] | да        | да     | да    | deadline-report × 1  |   1 | —    |
| missing-deadlines | [A3][run-023] | да        | да     | —     | 0                    |   1 | —    |
| missing-deadlines | [B3][run-024] | да        | да     | да    | deadline-report × 1  |   1 | —    |
| mcp-failure       | [A1][run-025] | да        | да     | —     | 0                    |   1 | E3   |
| mcp-failure       | [B1][run-026] | да        | да     | да    | open-task-export × 1 |   1 | E3   |
| mcp-failure       | [A2][run-027] | да        | да     | —     | 0                    |   1 | E3   |
| mcp-failure       | [B2][run-028] | да        | да     | да    | open-task-export × 1 |   1 | E3   |
| mcp-failure       | [A3][run-029] | да        | да     | —     | 0                    |   1 | E3   |
| mcp-failure       | [B3][run-030] | да        | да     | да    | open-task-export × 1 |   1 | E3   |

## Ошибки

- E1 · BUDGET_EXHAUSTED: side_effects limit exceeded; needs 1, remaining 0
- E2 · ERROR: POLICY_DENIED: Default permission for Bash
- E3 · ERROR: MCP_TOOL_ERROR: Tracker snapshot is unavailable or invalid

Гипотеза о росте качества подтвердилась на этой серии. В трёх сценариях отбора задач и дедлайнов число успешных результатов составило 8 для A и 9 для B.

## Пара A/B

- [deadlines A3][run-011], проверка не прошла. Запуск не завершился. Скиллы 0. Вызовы Bash × 3, list_files × 1, mcp__tracker__list_tasks × 1.
- [deadlines B3][run-012], проверка прошла. Результат совпал с эталоном. Скиллы deadline-report × 1. Вызовы list_files × 1, load_skill × 1, mcp__tracker__list_tasks × 1, write_file × 1.

[run-001]: trials/open-tasks/A1/events.jsonl "0c8bb40c6b964930a4d259ad72f1425e"
[run-002]: trials/open-tasks/B1/events.jsonl "48502ae0f02848b88bd4e6245929af22"
[run-003]: trials/open-tasks/A2/events.jsonl "664f335055944f4f9787e9777980b8c2"
[run-004]: trials/open-tasks/B2/events.jsonl "ec305a69ed234d2ea5ea9cfbbd230d84"
[run-005]: trials/open-tasks/A3/events.jsonl "e2e1cda5c83946da989e0d2989e400ba"
[run-006]: trials/open-tasks/B3/events.jsonl "a64322e43c464e82b2fbb33cbdca87c7"
[run-007]: trials/deadlines/A1/events.jsonl "603710f5bd1c41cc86e847c585a2cfee"
[run-008]: trials/deadlines/B1/events.jsonl "7b7860e1875f48df8996979303786437"
[run-009]: trials/deadlines/A2/events.jsonl "108934365d364fa4ac5ab82782b77ab9"
[run-010]: trials/deadlines/B2/events.jsonl "19a4086cff634c2982ab8e64013c0c4e"
[run-011]: trials/deadlines/A3/events.jsonl "16f02079049a4e0aa0f1a037d87ba51c"
[run-012]: trials/deadlines/B3/events.jsonl "d7055a80fb4840f080f49191a1bfb6b6"
[run-013]: trials/note/A1/events.jsonl "b6d98f7bb12c48adb3c52681362ff537"
[run-014]: trials/note/B1/events.jsonl "c8e7278fc0f2435596787c92629e1a02"
[run-015]: trials/note/A2/events.jsonl "25c642b4dfa24335bc128015209643e6"
[run-016]: trials/note/B2/events.jsonl "bc34b417309d415cbe48c123835b912d"
[run-017]: trials/note/A3/events.jsonl "f56a3bb365114c6084cddac16eebb7aa"
[run-018]: trials/note/B3/events.jsonl "3fb0d4b699cb4c93bea461be270039d9"
[run-019]: trials/missing-deadlines/A1/events.jsonl "5bf891a040cd403ba19ea348ba0f3fde"
[run-020]: trials/missing-deadlines/B1/events.jsonl "58db24f13b024fc48955b780135bc6f7"
[run-021]: trials/missing-deadlines/A2/events.jsonl "9bc463727b5f4cf78d1970f7bebb7380"
[run-022]: trials/missing-deadlines/B2/events.jsonl "2007b8044ea5436bab4e546c8b2ce1bc"
[run-023]: trials/missing-deadlines/A3/events.jsonl "09622bb238bc495781cf37d49c773b17"
[run-024]: trials/missing-deadlines/B3/events.jsonl "80b0455fcdc24dc1bb790cb6e058437a"
[run-025]: trials/mcp-failure/A1/events.jsonl "2ef77b93265549b6b494252a1277a8fb"
[run-026]: trials/mcp-failure/B1/events.jsonl "22a4dc97540c42d093b625611c766e2f"
[run-027]: trials/mcp-failure/A2/events.jsonl "3953ff39ee87474fb1a72dae670cbf83"
[run-028]: trials/mcp-failure/B2/events.jsonl "73e16f5c9e314306ab278db8eae84d04"
[run-029]: trials/mcp-failure/A3/events.jsonl "6010ecd6718b49459e1dbc1da74dab26"
[run-030]: trials/mcp-failure/B3/events.jsonl "ceb1775bfe044b6cb78028df05076523"
