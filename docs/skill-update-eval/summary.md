# Evals

Модель ollama/qwen3:4b-instruct. Сохранено попыток 30 из 30.

Новая версия сохраняет качество по всем сценариям и укладывается в лимит токенов.

Скилл `tracker-json-deadlines`. A · `1.0.0`. B · `1.1.0`.

## Результаты

| Сценарий          | A/B | Результат | Вызовы | Ошибки выбора | Инстр. | MCP | Сбои |
| ----------------- | --- | --------: | -----: | ------------: | -----: | --: | ---: |
| open-tasks        | A   |       0/3 |    3/3 |             3 |      6 |   3 |    0 |
| open-tasks        | B   |       0/3 |    3/3 |             3 |      6 |   3 |    0 |
| deadlines         | A   |       0/3 |    0/3 |             0 |      6 |   3 |    3 |
| deadlines         | B   |       0/3 |    3/3 |             0 |      9 |   3 |    0 |
| note              | A   |       3/3 |    3/3 |             0 |      3 |   0 |    0 |
| note              | B   |       3/3 |    3/3 |             0 |      3 |   0 |    0 |
| missing-deadlines | A   |       0/3 |    3/3 |             0 |      9 |   3 |    0 |
| missing-deadlines | B   |       0/3 |    3/3 |             0 |      9 |   3 |    0 |
| mcp-failure       | A   |       3/3 |    3/3 |             2 |      4 |   3 |    3 |
| mcp-failure       | B   |       3/3 |    3/3 |             3 |      3 |   3 |    3 |
| Итого             | A   |      6/15 |  12/15 |             5 |     28 |  12 |    6 |
| Итого             | B   |      6/15 |  15/15 |             6 |     30 |  12 |    3 |

В столбце «Вызовы» указано число попыток, прошедших проверку инструментов. В столбце «Сбои» учтены и ожидаемые ошибки MCP.

## Токены API

| Сценарий          | A/B |  Вход | Ср. вход | API вход | Выход | Ср. выход | API выход |
| ----------------- | --- | ----: | -------: | -------: | ----: | --------: | --------: |
| open-tasks        | A   | 16623 |   5541.0 |      9/9 |   411 |     137.0 |       9/9 |
| open-tasks        | B   | 16623 |   5541.0 |      9/9 |   456 |     152.0 |       9/9 |
| deadlines         | A   | 15486 |   5162.0 |      9/9 |  1287 |     429.0 |       9/9 |
| deadlines         | B   | 23079 |   7693.0 |    12/12 |  1023 |     341.0 |     12/12 |
| note              | A   |  7746 |   2582.0 |      6/6 |   156 |      52.0 |       6/6 |
| note              | B   |  7746 |   2582.0 |      6/6 |   156 |      52.0 |       6/6 |
| missing-deadlines | A   | 24636 |   8212.0 |    12/12 |   815 |     271.7 |     12/12 |
| missing-deadlines | B   | 24753 |   8251.0 |    12/12 |   819 |     273.0 |     12/12 |
| mcp-failure       | A   |  9877 |   3292.3 |      7/7 |   121 |      40.3 |       7/7 |
| mcp-failure       | B   |  7812 |   2604.0 |      6/6 |    99 |      33.0 |       6/6 |
| Итого             | A   | 74368 |   4957.9 |    43/43 |  2790 |     186.0 |     43/43 |
| Итого             | B   | 80013 |   5334.2 |    45/45 |  2553 |     170.2 |     45/45 |

Колонки API показывают долю обращений, для которых провайдер вернул usage. Если usage известен частично, в сумме только известные значения. Средние рассчитаны по попыткам с полным usage.

## Попытки

Номер попытки открывает журнал.

| Сценарий          | Попытка       | Результат | Вызовы | Выбор | Скиллы                     | MCP | Сбой |
| ----------------- | ------------- | --------- | ------ | ----- | -------------------------- | --: | ---- |
| open-tasks        | [A1][run-001] | нет       | да     | нет   | 0                          |   1 | —    |
| open-tasks        | [B1][run-002] | нет       | да     | нет   | 0                          |   1 | —    |
| open-tasks        | [A2][run-003] | нет       | да     | нет   | 0                          |   1 | —    |
| open-tasks        | [B2][run-004] | нет       | да     | нет   | 0                          |   1 | —    |
| open-tasks        | [A3][run-005] | нет       | да     | нет   | 0                          |   1 | —    |
| open-tasks        | [B3][run-006] | нет       | да     | нет   | 0                          |   1 | —    |
| deadlines         | [A1][run-007] | нет       | нет    | да    | tracker-json-deadlines × 1 |   1 | E1   |
| deadlines         | [B1][run-008] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| deadlines         | [A2][run-009] | нет       | нет    | да    | tracker-json-deadlines × 1 |   1 | E1   |
| deadlines         | [B2][run-010] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| deadlines         | [A3][run-011] | нет       | нет    | да    | tracker-json-deadlines × 1 |   1 | E1   |
| deadlines         | [B3][run-012] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| note              | [A1][run-013] | да        | да     | да    | 0                          |   0 | —    |
| note              | [B1][run-014] | да        | да     | да    | 0                          |   0 | —    |
| note              | [A2][run-015] | да        | да     | да    | 0                          |   0 | —    |
| note              | [B2][run-016] | да        | да     | да    | 0                          |   0 | —    |
| note              | [A3][run-017] | да        | да     | да    | 0                          |   0 | —    |
| note              | [B3][run-018] | да        | да     | да    | 0                          |   0 | —    |
| missing-deadlines | [A1][run-019] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| missing-deadlines | [B1][run-020] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| missing-deadlines | [A2][run-021] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| missing-deadlines | [B2][run-022] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| missing-deadlines | [A3][run-023] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| missing-deadlines | [B3][run-024] | нет       | да     | да    | tracker-json-deadlines × 1 |   1 | —    |
| mcp-failure       | [A1][run-025] | да        | да     | да    | open-task-export × 1       |   1 | E2   |
| mcp-failure       | [B1][run-026] | да        | да     | нет   | 0                          |   1 | E2   |
| mcp-failure       | [A2][run-027] | да        | да     | нет   | 0                          |   1 | E2   |
| mcp-failure       | [B2][run-028] | да        | да     | нет   | 0                          |   1 | E2   |
| mcp-failure       | [A3][run-029] | да        | да     | нет   | 0                          |   1 | E2   |
| mcp-failure       | [B3][run-030] | да        | да     | нет   | 0                          |   1 | E2   |

## Ошибки

- E1 · Model output was cut off; increase --max-tokens
- E2 · ERROR: MCP_TOOL_ERROR: Tracker snapshot is unavailable or invalid

## Пара A/B

- [open-tasks A1][run-001], проверка не прошла. Содержимое не совпало с эталоном. Скиллы 0. Вызовы mcp__tracker__list_tasks × 1, write_file × 1.
- [open-tasks B1][run-002], проверка не прошла. Содержимое не совпало с эталоном. Скиллы 0. Вызовы mcp__tracker__list_tasks × 1, write_file × 1.

[run-001]: trials/open-tasks/A1/events.jsonl "e1904778a2604225b16a4a3e95d8da3b"
[run-002]: trials/open-tasks/B1/events.jsonl "8c9c5e6addb044e8ac3c942dacc795b6"
[run-003]: trials/open-tasks/A2/events.jsonl "d29a3331d9b9438e9894effb58cf63bd"
[run-004]: trials/open-tasks/B2/events.jsonl "ba4b3c0367cb439189389f2e9e72042f"
[run-005]: trials/open-tasks/A3/events.jsonl "0dd01c9f67534999ac523365e42cf1a6"
[run-006]: trials/open-tasks/B3/events.jsonl "0ebf588f51df406581d87a886fc246da"
[run-007]: trials/deadlines/A1/events.jsonl "b437443950da466b9ceafd69dfe82fe1"
[run-008]: trials/deadlines/B1/events.jsonl "cb16475d615e41c5bde7c6033dcf13c4"
[run-009]: trials/deadlines/A2/events.jsonl "7093666099224d90987055c20b3861c4"
[run-010]: trials/deadlines/B2/events.jsonl "b6b5468ebf084cc5aedebd1da7ba0412"
[run-011]: trials/deadlines/A3/events.jsonl "201633359ee44f3991b0ccf4ca15242a"
[run-012]: trials/deadlines/B3/events.jsonl "21ee236f2ea340608222321f02763fac"
[run-013]: trials/note/A1/events.jsonl "83c4388a88b8433f8ef545bea41571d8"
[run-014]: trials/note/B1/events.jsonl "a975dbd9040741de89479fbe6db84794"
[run-015]: trials/note/A2/events.jsonl "615f194cbaf54f138cf027dd8c74c9f5"
[run-016]: trials/note/B2/events.jsonl "3c2829e7a428470cad7959a618bd7134"
[run-017]: trials/note/A3/events.jsonl "f0322a340a7d4ba891f2a85148f99595"
[run-018]: trials/note/B3/events.jsonl "15e753594e104b99aa373ce00e95318e"
[run-019]: trials/missing-deadlines/A1/events.jsonl "3ff8b2e1f32f4704997327a95efa81ff"
[run-020]: trials/missing-deadlines/B1/events.jsonl "33eae7ab8d084f8581459318c0af591c"
[run-021]: trials/missing-deadlines/A2/events.jsonl "274ff7c40ed24dbf8d386d2982595552"
[run-022]: trials/missing-deadlines/B2/events.jsonl "8706b59f541147278b92df26329ea273"
[run-023]: trials/missing-deadlines/A3/events.jsonl "dcca66dc3f4c473f868fa204b4576b51"
[run-024]: trials/missing-deadlines/B3/events.jsonl "25190b788b674f26b8585ac0928f3b44"
[run-025]: trials/mcp-failure/A1/events.jsonl "383462b65bfc494793b81fbe78772b52"
[run-026]: trials/mcp-failure/B1/events.jsonl "8ab629e6c0af45949634555d5a6317b3"
[run-027]: trials/mcp-failure/A2/events.jsonl "c4641f87b557472c93191d206c632a08"
[run-028]: trials/mcp-failure/B2/events.jsonl "9c6812fc397245f0942c08eadb8d8559"
[run-029]: trials/mcp-failure/A3/events.jsonl "1190c37c123949c0b82c06d040e9e523"
[run-030]: trials/mcp-failure/B3/events.jsonl "04c49ce4678340d8b4ceebe0b2dd8934"
