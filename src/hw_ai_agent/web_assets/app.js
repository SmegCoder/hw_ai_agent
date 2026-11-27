"use strict";
const $ = id => document.getElementById(id);
const esc = value =>
    String(value ?? "")
        .replace(
            /[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"}[c]));
const json = data => "<pre>" + esc(JSON.stringify(data, null, 2)) + "</pre>";
const info = text => "<p class=\"empty\">" + esc(text) + "</p>";
const button = (action, label, value = "", cls = "") => "<button class=\"" + cls + "\" data-action=\"" +
    action + "\" data-value=\"" + esc(value) + "\">" + esc(label) + "</button>";
const card = (name, html, wide = false) =>
    "<article class=\"card " + (wide ? "wide" : "") + "\"><h2>" + esc(name) + "</h2>" + html + "</article>";
const detail = (title, html) => "<details><summary>" + esc(title) + "</summary>" + html + "</details>";
const input = (id, label, val, hint = "") => "<label class=\"field\">" + esc(label) + "<input id=\"" + id +
    "\" value=\"" + esc(val) + "\" placeholder=\"" + esc(hint) + "\"></label>";
const check = (id, label, checked) => "<label class=\"check\"><input id=\"" + id + "\" type=\"checkbox\" " +
    (checked ? "checked" : "") + ">" + esc(label) + "</label>";
const select = (id, label, items, selected) => "<label class=\"field\">" + esc(label) + "<select id=\"" + id +
    "\">" +
    items
        .map(
            ([key, name]) => "<option value=\"" + esc(key) + "\" " + (key === selected ? "selected" : "") +
                ">" + esc(name) + "</option>")
        .join("") +
    "</select></label>";
const pages = {
  chat: "Диалог",
  preferences: "Настройки",
  settings: "Скиллы и MCP",
  doctor: "Доктор сессии",
  eval: "Eval и обновления"
};
let page = "chat", id = "", sessions = [], chat = null, prefs = null, currentReport = "", pageVersion = 0,
    lastChat = "", loading = false;
let reportRunning = false, submitBusy = false;
const systemTheme = matchMedia("(prefers-color-scheme: dark)");
function applyTheme() {
  const theme = $("theme").value;
  document.documentElement.dataset.theme =
      theme === "system" ? (systemTheme.matches ? "dark" : "light") : theme;
  localStorage.setItem("agi.theme", theme);
}
$("theme").value = localStorage.getItem("agi.theme") || "system";
$("theme").onchange = applyTheme;
systemTheme.addEventListener("change", applyTheme);
applyTheme();
for (const name of ["textSize", "spacing", "readingWidth", "showTurnStats", "wrapCode"]) {
  const field = $(name), key = "agi." + name, saved = localStorage.getItem(key);
  const checkbox = field.type === "checkbox";
  if (checkbox && saved !== null) field.checked = saved === "true";
  if (!checkbox && [...field.options].some(option => option.value === saved)) field.value = saved;
  function apply() {
    const value = checkbox ? String(field.checked) : field.value;
    document.documentElement.dataset[name] = value;
    localStorage.setItem(key, value);
  }
  field.onchange = apply;
  apply();
}
function notice(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
async function api(action, data = null, query = {}) {
  const url = "/api/" + action + "?" + new URLSearchParams(query);
  const options = data === null ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-Agi-UI": "1"},
    body: JSON.stringify(data)
  };
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw Error(value.error || response.statusText);
  return value.data;
}
const providerNames = {openrouter: "OpenRouter", openai: "OpenAI", ollama: "Ollama"};
let modelOptions;
function renderModelOptions(options) {
  modelOptions = options;
  const picker = $("newModel");
  const selected = picker.selectedOptions[0];
  const defaultLabel = "Из настроек · " + providerNames[options.provider] +
      (options.default_model ? " · " + options.default_model : "");
  picker.replaceChildren(new Option(defaultLabel, ""));
  for (const [provider, models] of Object.entries(options.models)) {
    const group = document.createElement("optgroup");
    group.label = providerNames[provider];
    for (const [value, name] of models) {
      const option = new Option(name, value);
      option.dataset.provider = provider;
      group.append(option);
      if (selected?.dataset.provider === provider && selected.value === value) option.selected = true;
    }
    picker.append(group);
  }
}
function modelSuggestions() {
  const models = modelOptions.models[$("provider").value];
  $("modelSuggestions").replaceChildren(...models.map(([value, label]) => new Option(label, value)));
}
function renderModelSettings(options) {
  renderModelOptions(options);
  $("provider").replaceChildren(...Object.keys(options.models).map(p => new Option(providerNames[p], p)));
  $("provider").value = options.provider;
  $("modelName").value = options.model;
  modelSuggestions();
}
$("provider").onchange = () => {
  $("modelName").value = "";
  modelSuggestions();
};
$("modelForm").onsubmit = guard(async event => {
  event.preventDefault();
  const options = await api("model", {provider: $("provider").value, model: $("modelName").value});
  renderModelSettings(options);
  notice("Модель для новых диалогов сохранена.");
});

let selectedFolder = null, browsing = ".", folderRoot = "", folderRequest = 0;
function chooseFolder(path) {
  selectedFolder = path;
  $("folderChoice").textContent = path === null ? "Новая папка с коротким ID" : path === "." ? folderRoot : path;
}
async function browseFolder(path) {
  const request = ++folderRequest;
  const result = await api("folders", null, {path});
  if (request !== folderRequest) return;
  browsing = result.path;
  folderRoot = result.root;
  $("folderLocation").textContent = browsing === "." ? folderRoot : folderRoot + "/" + browsing;
  $("parentFolder").disabled = browsing === ".";
  const list = $("folderList");
  list.replaceChildren();
  for (const name of result.folders) {
    const child = document.createElement("button");
    child.textContent = name + " ›";
    child.onclick = guard(() => browseFolder(browsing === "." ? name : browsing + "/" + name));
    list.append(child);
  }
  if (!result.folders.length) list.textContent = "Вложенных папок нет";
}
$("chooseFolder").onclick = guard(async () => {
  await browseFolder(selectedFolder || ".");
  $("folderDialog").showModal();
});
$("parentFolder").onclick = guard(() => browseFolder(browsing.split("/").slice(0, -1).join("/") || "."));
$("useFolder").onclick = () => { chooseFolder(browsing); $("folderDialog").close(); };
$("autoFolder").onclick = () => { chooseFolder(null); $("folderDialog").close(); };
$("cancelFolder").onclick = () => $("folderDialog").close();

function setEventsOpen(open) {
  $("chatPage").classList.toggle("collapsed", !open);
  $("rightDivider").hidden = $("inspector").hidden = !open;
  $("toggleEvents").setAttribute("aria-expanded", String(open));
  $("toggleEvents").textContent = open ? "Скрыть панель" : "Показать панель";
  localStorage.setItem("agi.events", open ? "open" : "closed");
}
setEventsOpen(localStorage.getItem("agi.events") !== "closed");
$("toggleEvents").onclick = () => setEventsOpen($("inspector").hidden);

function setupDivider(side, pane) {
  const handle = $(side + "Divider");
  const property = "--" + side + "-width", key = "agi." + side + "Width";
  const left = side === "left";
  const container = handle.parentElement;
  const minimum = left ? 150 : 180;
  const maximum = () => Math.max(minimum, Math.floor(container.clientWidth * (left ? .32 : .46)));
  const width = () => Math.round(pane.getBoundingClientRect().width);
  function updateAria() {
    if (!pane.clientWidth || !handle.offsetWidth) return;
    handle.setAttribute("aria-valuemin", minimum);
    handle.setAttribute("aria-valuemax", maximum());
    handle.setAttribute("aria-valuenow", width());
    handle.setAttribute("aria-valuetext", width() + " пикселей");
  }
  function resize(requested) {
    const px = Math.max(minimum, Math.min(maximum(), requested));
    container.style.setProperty(property, px + "px");
    updateAria();
  }
  const saved = Number(localStorage.getItem(key));
  if (saved >= minimum) container.style.setProperty(property, saved + "px");
  new ResizeObserver(updateAria).observe(pane);
  let origin = 0, initial = 0;
  handle.onpointerdown = event => {
    if (event.button !== 0 || !event.isPrimary) return;
    origin = event.clientX;
    initial = width();
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("dragging");
    event.preventDefault();
  };
  handle.onpointermove = event => {
    if (handle.hasPointerCapture(event.pointerId))
      resize(initial + (event.clientX - origin) * (left ? 1 : -1));
  };
  handle.onlostpointercapture = () => {
    handle.classList.remove("dragging");
    localStorage.setItem(key, width());
  };
  handle.onkeydown = event => {
    const targets = {ArrowLeft: width() + (left ? -16 : 16), ArrowRight: width() + (left ? 16 : -16),
                     Home: minimum, End: maximum()};
    if (!(event.key in targets)) return;
    event.preventDefault();
    resize(targets[event.key]);
    localStorage.setItem(key, width());
  };
  return () => { container.style.removeProperty(property); localStorage.removeItem(key); };
}
const resetLeft = setupDivider("left", $("sidebar"));
const resetRight = setupDivider("right", $("inspector"));
$("resetLayout").onclick = () => { resetLeft(); resetRight(); notice("Размеры панелей сброшены."); };

async function listSessions() {
  sessions = await api("sessions");
  renderSessions();
}
function renderSessions() {
  $("sessions").innerHTML =
      sessions.map(s => button("session", s.title || "Новый диалог", s.id, s.id === id ? "active" : ""))
          .join("") ||
      info("Пока нет диалогов");
}
const labels = {
  skill_loaded: "Загружен скилл",
  skills_discovered: "Доступные скиллы",
  mcp_discovered: "Подключён MCP",
  mcp_error: "Ошибка MCP",
  tool_call: "Вызов инструмента",
  tool_result: "Результат инструмента",
  model_request: "Запрос к модели",
  model_usage: "Расход токенов",
  policy_decision: "Решение policy",
  session_created: "Запуск создан",
  session_completed: "Запуск завершён"
};
function eventHtml(e) {
  const data = e.data || {};
  const time =
      new Date(e.time).toLocaleTimeString("ru", {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  return "<details class=\"event\" data-event=\"" + esc(e.id) + "\"><summary>" +
      esc(labels[e.type] || e.type) + "<small>" +
      esc([time, data.name || data.tool, data.outcome].filter(Boolean).join(" · ")) + "</small></summary>" +
      info(e.id) + json(data) + "</details>";
}
function renderEvents() {
  const opened = new Set([...$("events").querySelectorAll("details[open]")].map(n => n.dataset.event));
  const filter = $("filter").value;
  const events = (chat?.events || []).filter(e => {
    const key = [
      e.type, e.data?.name, e.data?.server ? "mcp" : "", e.data?.error || e.data?.failed ? "error" : "", e.data?.outcome,
      e.data?.action
    ].join(" ");
    return filter === "error" ? /error|refused|blocked|deny|failed/i.test(key) : key.includes(filter);
  });
  $("eventCount").textContent = events.length;
  $("events").innerHTML =
      events.slice().reverse().map(eventHtml).join("") || info("Здесь появятся действия агента.");
  for (const item of $("events").children)
    if (opened.has(item.dataset.event)) item.open = true;
}
let inspectorPage = "events", folder = ".", filesRequest = 0, previewRequest = 0;
function selectInspector(section) {
  inspectorPage = section;
  $("eventPanel").hidden = section !== "events";
  $("filesPanel").hidden = section !== "files";
  $("eventsTab").setAttribute("aria-pressed", String(section === "events"));
  $("filesTab").setAttribute("aria-pressed", String(section === "files"));
  if (section === "files") loadFiles();
}
function resetFiles() {
  folder = ".";
  filesRequest++;
  previewRequest++;
  if ($("previewDialog").open) $("previewDialog").close();
  $("chatFiles").replaceChildren();
  $("fileChanges").replaceChildren();
  $("fileDirectory").replaceChildren();
}
function fileButton(label, click) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = label;
  b.onclick = click;
  return b;
}
async function loadFiles() {
  const request = ++filesRequest, key = id, directory = folder;
  const listing = $("chatFiles"), changes = $("fileChanges");
  $("fileDirectory").textContent = directory === "." ? "/" : "/" + directory;
  listing.replaceChildren();
  changes.replaceChildren();
  if (directory !== ".") listing.append(fileButton("↑ На уровень выше", () => {
    folder = directory.split("/").slice(0, -1).join("/") || ".";
    loadFiles();
  }));
  const message = document.createElement("p");
  message.className = "empty";
  message.textContent = key ? "Загрузка…" : "Файлы появятся после создания чата.";
  listing.append(message);
  if (!key) return;
  try {
    const data = await api("chat-files", null, {id: key, path: directory});
    if (request !== filesRequest || id !== key) return;
    message.textContent = "Папка пуста";
    if (data.items.length) message.remove();
    for (const item of data.items) {
      const path = directory === "." ? item.name : directory + "/" + item.name;
      listing.append(fileButton(item.name + (item.directory ? "/" : ""), () => {
        if (item.directory) { folder = path; loadFiles(); }
        else previewFile(path);
      }));
    }
    if (data.changes.length) {
      const h = document.createElement("h3");
      h.textContent = "Изменения по запросам";
      changes.append(h);
      for (const change of data.changes) {
        const label = "№" + change.turn + " · " + change.path + (change.rolled_back ? " · отменено" : "");
        changes.append(fileButton(label, () => previewFile(change.path, change.run, change.turn)));
      }
    }
  } catch (error) {
    if (request === filesRequest) message.textContent = error.message;
  }
}
async function previewFile(path, run, turn) {
  const request = ++previewRequest, key = id;
  $("previewTitle").textContent = run ? "Изменения · №" + turn + " · " + path : path;
  $("previewBody").textContent = "Загрузка…";
  $("previewDialog").showModal();
  try {
    const query = {id: key, path};
    if (run) query.run = run;
    const result = await api("chat-file", null, query);
    if (request !== previewRequest || key !== id) return;
    const pre = document.createElement("pre");
    pre.textContent = result.text;
    if (!result.text) pre.textContent = "Пустой файл";
    if (run) {
      const lines = document.createDocumentFragment();
      for (const line of result.text.split("\n")) {
        const row = document.createElement("span");
        row.className = line.startsWith("+") ? "added" : line.startsWith("-") ? "removed" : "";
        row.textContent = line + "\n";
        lines.append(row);
      }
      pre.replaceChildren(lines);
    }
    $("previewBody").replaceChildren(pre);
  } catch (error) {
    if (request === previewRequest) $("previewBody").textContent = error.message;
  }
}
$("filesTab").onclick = () => selectInspector("files");
$("eventsTab").onclick = () => selectInspector("events");
$("reloadFiles").onclick = loadFiles;
$("closePreview").onclick = () => $("previewDialog").close();

function progressText(state) {
  if (state.approval) return "Ждёт подтверждения · " + (state.approval.target || state.approval.tool);
  const event = state.active_run && state.events?.findLast(e => e.run_id === state.active_run &&
      ["tool_call", "tool_result", "model_request", "model_usage"].includes(e.type));
  if (!event) return "Подготавливает запрос…";
  if (event.type === "model_request") return "Ожидает ответ модели…";
  if (event.type === "model_usage") return "Обрабатывает ответ модели…";
  const data = event.data;
  if (event.type === "tool_result") return data.failed ? "Обрабатывает ошибку инструмента…" : "Обрабатывает результат инструмента…";
  let path;
  try { path = JSON.parse(data.arguments).path; } catch {}
  const actions = {read_file: "Читает файл", write_file: "Записывает файл", delete_file: "Удаляет файл",
                   list_files: "Просматривает папку", mcp__tracker__list_tasks: "Читает задачи трекера"};
  return (Object.hasOwn(actions, data.name) ? actions[data.name] : "Вызывает " + data.name) + (typeof path === "string" ? " · " + path : "…");
}
function tokens(x) {
  if (!x) return "—";
  if (x.tokens === null) return "неизвестно";
  return x.tokens.toLocaleString("ru") + (x.reported_requests < x.requests ? " (частично)" : "");
}
function metrics(s) {
  if (!s) return info("Расход появится после запуска.");
  return "<div class=\"metrics\">" +
      [["Вход", tokens(s.usage.prompt_tokens)], ["Выход", tokens(s.usage.completion_tokens)]]
          .map(([k, v]) => "<div class=\"metric\"><small>" + k + "</small><b>" + esc(v) + "</b></div>")
          .join("") +
      "</div>" + detail("Вызовы · " + Object.values(s.tool_calls).reduce((a, b) => a + b, 0), json({
                          модель: s.model_requests,
                          инструменты: s.tool_calls,
                          MCP: s.mcp_calls,
                          скиллы: s.skill_loads
                        }));
}
function inlineAnswer(text) {
  const tokens = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*|(?<![\w\p{L}])_[^_\n]+_(?![\w\p{L}]))/gu;
  return text.split(tokens).map((part, index) => {
    if (index % 2 === 0) return esc(part);
    if (part.startsWith("`")) return "<code>" + esc(part.slice(1, -1)) + "</code>";
    if (part.startsWith("**")) return "<strong>" + esc(part.slice(2, -2)) + "</strong>";
    return "<em>" + esc(part.slice(1, -1)) + "</em>";
  }).join("");
}
function answerHtml(answer) {
  const blocks = [], lines = answer.replace(/\r\n?/g, "\n").split("\n");
  let paragraph = [];
  function flush() {
    if (paragraph.length) blocks.push("<p>" + inlineAnswer(paragraph.join("\n")) + "</p>");
    paragraph = [];
  }
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i], fence = line.match(/^ {0,3}(`{3,}|~{3,})\s*([^\s]*)\s*$/);
    const heading = line.match(/^#{1,6}\s+(.+)/);
    const item = /^\s*(?:([-+*])|(\d{1,9})\.)\s+(.+)/;
    const first = line.match(item);
    if (fence) {
      flush();
      const code = [], end = new RegExp("^ {0,3}" + fence[1][0] + "{" + fence[1].length + ",}\\s*$");
      while (++i < lines.length && !end.test(lines[i])) code.push(lines[i]);
      blocks.push("<pre class=\"codeBlock\"><code>" + esc(code.join("\n")) + "</code></pre>");
    } else if (heading) {
      flush();
      blocks.push("<h3>" + inlineAnswer(heading[1]) + "</h3>");
    } else if (first) {
      flush();
      const ordered = !!first[2], tag = ordered ? "ol" : "ul";
      const items = ["<li>" + inlineAnswer(first[3]) + "</li>"];
      while (i + 1 < lines.length) {
        const next = lines[i + 1].match(item);
        if (!next || !!next[2] !== ordered) break;
        items.push("<li>" + inlineAnswer(next[3]) + "</li>");
        i++;
      }
      const start = ordered ? ' start="' + Number(first[2]) + '"' : "";
      blocks.push("<" + tag + start + ">" + items.join("") + "</" + tag + ">");
    } else if (!line.trim()) flush();
    else paragraph.push(line);
  }
  flush();
  return blocks.join("");
}
const runLabels = {
  completed: "Готово", running: "В работе", ready: "Готов к запуску", failed: "Ошибка",
  interrupted: "Прерван", blocked: "Заблокирован", aborted: "Отменён", rolled_back: "Изменения отменены",
  waiting_approval: "Ожидает решения", waiting_plan_approval: "Ожидает решения по плану",
  budget_exhausted: "Лимит исчерпан", context_limit: "Достигнут предел контекста", journal_conflict: "Конфликт правок"
};
function exchangeHtml(turn, opened, progress = "") {
  const key = esc(turn.id);
  const answer = turn.answer ? '<div class="answer" aria-label="Ответ агента">' + answerHtml(turn.answer) + '</div>' : "";
  const copy = turn.answer ? '<button type="button" class="copyAnswer" data-copy="' + key +
      '" aria-label="Копировать ответ" title="Копировать ответ"><span aria-hidden="true"></span></button>' : "";
  const status = esc(runLabels[turn.status] || turn.status);
  return '<article class="exchange"><div class="bubble" aria-label="Ваше сообщение">' + esc(turn.prompt) +
       '</div>' + (progress ? '<div class="turnProgress" role="status">' + esc(progress) + '</div>' : '') +
      answer + '<div class="answerActions">' + copy +
      '<details class="turnStats" data-turn="' + key + '"' + (opened.has(turn.id) ? " open" : "") +
      '><summary>Статистика<span> · ' + status + '</span></summary>' + metrics(turn.stats) + '</details></div></article>';
}
function renderChat() {
  const turns = chat?.turns || [];
  const atBottom =
      $("conversation").scrollHeight - $("conversation").scrollTop - $("conversation").clientHeight < 45;
  $("title").textContent = turns.length ? turns[0].prompt.slice(0, 75) : "Новый диалог";
  $("model").textContent = chat?.model || "";
  $("workspacePath").textContent = chat?.workspace || "";
  $("workspacePath").title = chat?.workspace || "";
  $("newModelPicker").hidden = Boolean(id);
  $("deleteChat").hidden = !id;
  $("deleteChat").disabled = submitBusy || !chat || chat.running;
  const pending = chat?.approval;
  const openStats = new Set([...$("messages").querySelectorAll(".turnStats[open]")].map(el => el.dataset.turn));
  const progress = chat?.running ? progressText(chat) : "";
  const inTurn = progress && turns.some(t => t.id === chat.active_run);
  $("messages").innerHTML = turns.length || progress ? turns.map(t =>
      exchangeHtml(t, openStats, t.id === chat.active_run ? progress : "")).join("") +
      (progress && !inTurn ? (chat.starting_prompt && turns.at(-1)?.prompt !== chat.starting_prompt ?
          '<div class="bubble">' + esc(chat.starting_prompt) + '</div>' : '') +
          '<div class="turnProgress" role="status">' + esc(progress) + '</div>' : '') :
      "<div class=\"intro\"><h2>С чем помочь по учёбе?</h2><p>Спроси о задачах трекера или попроси сохранить отчёт.</p>" +
          button("suggest", "Открытые задачи", "Какие задачи открыты в учебном трекере?") +
          button(
              "suggest", "Отчёт по срокам",
              "Собери отчёт о просроченных задачах на 25 ноября 2026 года и сохрани его в deadlines.md.") +
          "</div>";
  $("approval").innerHTML = pending ?
      "<div class=\"approval\"><small>Требуется подтверждение</small><h2>" + esc(pending.tool) + " · " +
          esc(pending.target) + "</h2><p>" + esc(pending.reason) + "</p>" + jsonOrText(pending.preview) +
          "<div class=\"buttons\">" + button("allow", "Разрешить", pending.id, "primary") +
          button("deny", "Отклонить", pending.id) + "</div></div>" :
      "";
  $("runStatus").innerHTML = esc(chat?.error ||
                                 (chat?.ended       ? "Диалог завершён" :

                                                      "")) +
      (chat?.pending && !chat.running && !chat.ended ? button("resume", "Продолжить запрос") : "");
  $("send").disabled = submitBusy || !!(chat?.running || chat?.ended || chat?.pending);
  $("prompt").disabled = !!chat?.ended;
  $("stats").innerHTML = "<h2>Расход сессии</h2>" + metrics(chat?.stats);
  renderEvents();
  if (atBottom || pending) $("conversation").scrollTop = $("conversation").scrollHeight;
}
const jsonOrText = s => s ? "<pre>" + esc(s) + "</pre>" : "";
async function refreshChat() {
  const key = id, result = key ? await api("chat", null, {id: key}) : null;
  if (key !== id || page !== "chat") return;
  const sig = JSON.stringify(result);
  if (lastChat !== sig) {
    const needFiles = !lastChat || chat?.running !== result?.running || chat?.events?.length !== result?.events?.length;
    chat = result;
    if (inspectorPage === "files" && needFiles) loadFiles();
    lastChat = sig;
    renderChat();
  }
}
async function show(target) {
  const version = ++pageVersion;
  page = target;
  notice();
  $("crumb").textContent = pages[page];
  document.querySelectorAll("[data-page]")
      .forEach(b => b.classList.toggle("active", b.dataset.page === page));
  $("chatPage").hidden = page !== "chat";
  $("secondary").hidden = page === "chat" || page === "preferences";
  $("preferences").hidden = page !== "preferences";
  $("toggleEvents").hidden = page !== "chat";
  if (page === "preferences") {
    const options = await api("chat-options");
    if (version === pageVersion) renderModelSettings(options);
    return;
  }
  if (page === "chat") {
    lastChat = "";
    await refreshChat();
    return;
  }
  $("secondary").innerHTML = "<h1>" + esc(pages[page]) + "</h1>" + info("Загрузка…");
  if (page === "settings") {
    const data = await api("settings");
    if (version !== pageVersion) return;
    prefs = data;
    renderSettings();
  } else if (page === "doctor") {
    const data = id ? await api("doctor", null, {id}) : null;
    if (version !== pageVersion) return;
    $("secondary").innerHTML =
        "<h1>Доктор сессии</h1>" + (!data ? info("Открой диалог в списке сессий.") : doctorHtml(data));
  } else
    await loadReports(version);
}
function renderSettings() {
  const p = prefs, flags = p.sources[p.source] || {check_on_start: true, auto_update: true};
  const installed =
      p.skills
          .map(
              s => "<label class=\"check listrow\"><input type=\"checkbox\" data-skill=\"" + esc(s.name) +
                  "\" " + (!p.disabled.includes(s.name) ? "checked" : "") + "><span><b>" + esc(s.name) +
                  "</b><br><small>" + esc(s.origin + " · " + s.description) + "</small></span></label>")
          .join("");
  $("secondary").innerHTML = "<h1>Скиллы и MCP</h1>" +
      info("Настройки диалога применяются при создании новой сессии.") + "<div class=\"cards\">" +
      card("MCP-серверы",
           input("mcpPath", "Путь к конфигурации MCP", p.mcp, "mcp/tracker.json") +
               button("mcp", "Инструменты подключения") + "<div id=\"mcpResult\"></div>") +
      card("Скиллы",
           input("skillsDir", "Дополнительная папка SKILL.md", p.skills_dir, "Необязательно") +
               check("demo", "Только встроенные демо-скиллы", p.demo) + installed) +
      card("Маркет",
           input("source", "Источник", p.source, "Путь к catalog.json") + "<div class=\"buttons\">" +
               button("source", "Учебный каталог", "demo") + button("source", "Anthropic", "anthropic") +
               "</div>" + check("checkStart", "Проверять при старте", flags.check_on_start) +
               check("autoUpdate", "Автообновление с проверкой через eval", flags.auto_update) +
               info("Автообновление создаёт кандидата. До применения сравниваются обе версии.") +
               (p.demo ? info("Демо-режим: скиллы маркета недоступны новым диалогам.") : "") +
               (!p.market_install_allowed ? info("Policy запрещает install_skill: агент не увидит предложения маркета. " +
                   "Установленные скиллы можно загружать, если разрешён load_skill. Policy: " + p.policy) : "")) +
      "</div><div class=\"buttons\">" + button("save", "Сохранить настройки", "", "primary") +
      button("check", "Сохранить и проверить источник") +
      (p.source === "demo" ? button("publish", "Опубликовать учебную v1.1", "1.1.0") : "") +
      "</div><h2 class=\"subhead\">Каталог</h2><input id=\"search\" aria-label=\"Найти скилл\" placeholder=\"Поиск по каталогу\"><div id=\"offers\"></div>" +
      detail("Новинки и журнал маркета",
             p.events.slice()
                     .reverse()
                     .map(
                         e => "<div class=\"listrow\">" +
                             esc([e.event, e.name, e.status, e.error].filter(Boolean).join(" · ")) + "</div>")
                     .join("") ||
                 info("Проверок ещё нет."));
  $("source").onchange = () => {
    const f = p.sources[$("source").value.trim()] || {check_on_start: true, auto_update: true};
    $("checkStart").checked = f.check_on_start;
    $("autoUpdate").checked = f.auto_update;
  };
  $("search").oninput = renderOffers;
  renderOffers();
}
function renderOffers() {
  const term = $("search").value.toLowerCase();
  $("offers").innerHTML =
      prefs.offers.filter(o => (o.name + " " + o.description).toLowerCase().includes(term))
          .map(
              o => "<div class=\"listrow\"><b>" + esc(o.name) + "</b> <small>" + esc(o.version) +
                  "</small> " +
                  (prefs.skills.some(s => s.name === o.name) ? "Установлен" :
                                                               button("install", "Установить", o.name)) +
                  "<p>" + esc(o.description) + "</p></div>")
          .join("") ||
      info("Нет карточек. Проверь источник или измени поиск.");
}
async function saveSkillSettings() {
  const disabled = new Set(prefs.disabled);
  document.querySelectorAll("[data-skill]")
      .forEach(el => el.checked ? disabled.delete(el.dataset.skill) : disabled.add(el.dataset.skill));
  return api("settings", {
    mcp: $("mcpPath").value,
    source: $("source").value,
    demo: $("demo").checked,
    skills_dir: $("skillsDir").value,
    check_on_start: $("checkStart").checked,
    auto_update: $("autoUpdate").checked,
    disabled: [...disabled]
  });
}
function doctorHtml(d) {
  return "<div class=\"cards\">" + card("Расход диалога", metrics(d.total)) +
      card("Расход доктора", info("Запросов к модели: " + d.doctor_usage.model_requests)) + "</div>" +
      d.runs
          .map(
              (r, i) => card(
                  "Запрос " + (i + 1) + " · " + r.status,
                  metrics(r) +
                      (r.findings
                           .map(
                               f => "<div class=\"listrow\"><b>" + esc(f.tool + " × " + f.count) + "</b><p>" +
                                   esc(f.interpretation) + "</p><p>" + esc(f.recommendation) + "</p>" +
                                   f.event_ids
                                       .map(e => button("event", "Событие " + e.split(":").pop(), e, "link"))
                                       .join(" ") +
                                   "</div>")
                           .join("") ||
                       info("Доктор не обнаружил повторяющихся проблем.")) +
                      (r.limitations || []).map(info).join("")))
          .join("");
}
const archiveTitles = {
  "archive-ab": "Сохранённый A/B · учебный трекер",
  "archive-update": "Сохранённое сравнение версий"
};
async function loadReports(version) {
  const [index, p] = await Promise.all([api("reports"), api("settings")]);
  if (version !== pageVersion) return;
  reportRunning = index.running;
  const names = index.names;
  if (!names.includes(currentReport) && !index.running) currentReport = names.at(-1) || "";
  const r = names.includes(currentReport) ? await api("report", null, {key: currentReport}) : null;
  if (version !== pageVersion) return;
  const candidates =
      p.candidates.slice()
          .reverse()
          .map(
              c => "<div class=\"listrow\"><b>" + esc(c.name) + "</b><p>" +
                  esc(c.old_version + " → " + c.new_version + " · " + c.status) + "</p>" +
                  (c.status === "pending" ? button("candidate", "Сравнить версии", c.name) : "") +
                  (c.status === "accepted" ? button("rollback", "Откатить обновление", c.name) : "") +
                  (c.decision ? detail("Причины решения", json(c.decision.reasons)) : "") + "</div>")
          .join("");
  $("secondary").innerHTML = "<h1>Eval и обновления</h1><div class=\"cards\">" +
      card("Проверка агента",
           info("5 сценариев × 3 попытки × 2 версии. Для каждого варианта создаются независимые попытки.") +
               button("eval", "Новая серия", "", "primary") +
               (currentReport.startsWith("ab/") && (r?.summary?.completed ?? 0) < 30 ?
                    button("continueEval", "Продолжить серию", currentReport) :
                    "") +
               info(index.error || (index.running ? "Выполняется серия…" : ""))) +
      card("Кандидаты обновлений",
           (p.candidates.some(c => c.status === "pending") ?
                select(
                    "focus", "Явно загрузить скилл в сценарии",
                    [["", "Без указания скилла"], ...p.cases.map(c => [c, c])], "") :
                "") +
               (candidates ||
                info("После проверки маркета здесь появятся новые версии установленных скиллов."))) +
      "</div>" + select("report", "Отчёт", names.map(n => [n, archiveTitles[n] || n]), currentReport) +
      (r ? renderReport(r) : info("Нет сохранённых серий."));
  $("report").onchange = async () => {
    currentReport = $("report").value;
    await show("eval");
  };
}
function renderReport(r) {
  const s = r.summary;
  const rows = (thead, body) => "<div class=\"tableWrap\"><table><thead><tr>" +
      thead.map(x => "<th>" + x + "</th>").join("") + "</tr></thead><tbody>" + body +
      "</tbody></table></div>";
  const counts = v => v ? esc(v.outcome_pass + "/" + v.attempts) : "—";
  const cases = Object.entries(s.by_case || {})
                    .map(
                        ([name, v]) => "<tr><td>" + esc(name) + "</td><td>" + counts(v.A) + "</td><td>" +
                            counts(v.B) + "</td></tr>")
                    .join("");
  const trials =
      r.rows
          .map(
              t => "<tr><td>" + esc(t.case_id) + "</td><td>" + esc(t.variant + t.trial) + "</td><td>" +
                  esc(t.check || t.status) + "</td><td>" + button("trial", "Журнал", t.run_id) + "</td></tr>")
          .join("");
  return "<h2 class=\"subhead\">Результаты · " + esc(s.completed ?? 0) + " / " + esc(s.planned ?? 30) +
      "</h2>" +
      (r.decision ? card("Решение · " + r.decision.status, esc(r.decision.reasons.join(" "))) : "") +
      rows(["Сценарий", "Успешно A", "Успешно B"], cases) +
      detail("Токены, маршрутизация и вызовы по версиям", json(s.by_variant || {})) +
      "<h2 class=\"subhead\">Попытки</h2>" + rows(["Сценарий", "Попытка", "Результат", "События"], trials) +
      "<div id=\"trial\"></div>";
}
async function submit() {
  const prompt = $("prompt").value.trim();
  if (!prompt || submitBusy) return;
  submitBusy = true;
  $("send").disabled = true;
  try {
    if (!id) {
      const selected = $("newModel").selectedOptions[0];
      const choice = selected.value ? {provider: selected.dataset.provider, model: selected.value} : {};
      if (selectedFolder !== null) choice.folder = selectedFolder;
      id = await api("new", choice);
      localStorage.setItem("agi.chat", id);
    }
    await api("send", {id, prompt});
    $("prompt").value = "";
    await listSessions();
    await refreshChat();
  } finally {
    submitBusy = false;
    renderChat();
  }
}
async function act(action, v) {
  notice();
  if (action === "session" || action === "new") {
    if (submitBusy) return;
    resetFiles();
    id = action === "session" ? v : "";
    if (!id) {
      $("newModel").value = "";
      chooseFolder(null);
    }
    chat = null;
    localStorage.setItem("agi.chat", id);
    $("prompt").value = "";
    await listSessions();
    await show("chat");
  } else if (action === "suggest") {
    $("prompt").value = v;
    $("prompt").focus();
  } else if (action === "allow" || action === "deny") {
    await api("decision", {id, request: v, allow: action === "allow"});
    await refreshChat();
  } else if (action === "resume") {
    await api("send", {id, prompt: "", resume: true});
    await refreshChat();
  } else if (action === "event") {
    await show("chat");
    setEventsOpen(true);
    selectInspector("events");
    $("filter").value = "";
    renderEvents();
    const el = [...$("events").children].find(x => x.dataset.event === v);
    if (el) {
      el.open = true;
      el.scrollIntoView({block: "nearest"});
    }
  } else if (action === "source") {
    $("source").value = v;
    $("source").onchange();
  } else if (action === "publish") {
    await api("publish-demo", {version: v});
    notice("В учебном каталоге опубликована v1.1. Проверь источник, чтобы увидеть кандидата.");
  } else if (action === "save" || action === "check") {
    prefs = await saveSkillSettings();
    let error = null;
    if (action === "check") {
      try {
        await api("check", {});
      } catch (exc) {
        error = exc;
      }
    }
    await show("settings");
    if (error) throw error;
    notice(action === "check" ? "Каталог проверен." : "Настройки сохранены.");
  } else if (action === "mcp") {
    const data = await api("mcp");
    $("mcpResult").innerHTML =
        data.tools
                .map(t => detail(t.function.name, esc(t.function.description) + json(t.function.parameters)))
                .join("") +
            data.events.filter(e => e.error).map(e => info(e.error)).join("") ||
        info("Нет подключённых инструментов. Сохрани путь к конфигурации.");
  } else if (action === "install") {
    await api("install", {name: v});
    await show("settings");
    notice("Скилл установлен.");
  } else if (["eval", "continueEval", "candidate"].includes(action)) {
    const focus = $("focus")?.value;
    const series = await api("eval", {
      candidate: action === "candidate" ? v : null,
      focus: focus ? [focus] : [],
      series: action === "continueEval" ? v : null
    });
    if (series) currentReport = series;
    await show("eval");
  } else if (action === "rollback") {
    await api("rollback", {name: v});
    await show("eval");
    notice("Восстановлена предыдущая версия.");
  } else if (action === "trial") {
    const events = await api("report", null, {key: currentReport, run: v});
    $("trial").innerHTML = "<h2 class=\"subhead\">Журнал попытки</h2>" + events.map(eventHtml).join("");
    $("trial").scrollIntoView();
  }
}
function guard(fn) {
  return async event => {
    try {
      await fn(event);
    } catch (error) {
      notice(error.message);
    }
  };
}
document.addEventListener("click", guard(async e => {
                            const b = e.target.closest("button");
                            if (!b) return;
                            if (b.dataset.page) await show(b.dataset.page);
                            if (b.dataset.action) {
                              b.disabled = true;
                              try {
                                await act(b.dataset.action, b.dataset.value);
                              } finally {
                                b.disabled = false;
                              }
                            }
                          }));
$("messages").addEventListener("click", guard(async event => {
  const control = event.target.closest("[data-copy]");
  if (!control) return;
  const turn = chat?.turns.find(t => t.id === control.dataset.copy);
  if (!turn?.answer) return;
  await navigator.clipboard.writeText(turn.answer);
  control.classList.add("copied");
  control.title = control.ariaLabel = "Скопировано";
  setTimeout(() => {
    control.classList.remove("copied");
    control.title = control.ariaLabel = "Копировать ответ";
  }, 1600);
}));
$("deleteChat").onclick = () => {
  if (!id || !chat || submitBusy || chat.running) return;
  $("deleteDialog").dataset.chat = id;
  $("deleteName").textContent = $("title").textContent;
  $("deleteError").hidden = true;
  $("deleteDialog").showModal();
};
$("cancelDelete").onclick = () => $("deleteDialog").close();
$("deleteDialog").oncancel = event => {
  if ($("confirmDelete").disabled) event.preventDefault();
};
$("confirmDelete").onclick = guard(async () => {
  const target = $("deleteDialog").dataset.chat;
  $("confirmDelete").disabled = $("cancelDelete").disabled = true;
  try {
    await api("delete-chat", {id: target});
  } catch (error) {
    $("deleteError").textContent = error.message;
    $("deleteError").hidden = false;
    return;
  } finally {
    $("confirmDelete").disabled = $("cancelDelete").disabled = false;
  }
  $("deleteDialog").close();
  if (id === target) await act("new", "");
  else await listSessions();
  notice("Чат удалён.");
});
$("new").onclick = guard(() => act("new", ""));
$("filter").onchange = renderEvents;
$("compose").onsubmit = guard(async e => {
  e.preventDefault();
  notice();
  await submit();
});
$("prompt").onkeydown = e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("compose").requestSubmit();
  }
};
async function refresh() {
  if (loading || document.hidden) return;
  loading = true;
  try {
    if (page === "chat" && id && (chat?.running || chat?.pending || !chat)) {
      const previous = chat?.running;
      await refreshChat();
      if (previous && !chat?.running) await listSessions();
    }
    if (page === "eval" && reportRunning) await loadReports(pageVersion);
  } catch (e) {
    notice(e.message);
  } finally {
    loading = false;
  }
}
(async () => {
  try {
    await Promise.all([listSessions(), api("chat-options").then(renderModelOptions)]);
    const saved = localStorage.getItem("agi.chat");
    if (sessions.some(s => s.id === saved)) id = saved;
    renderSessions();
    await show("chat");
    setInterval(refresh, 1100);
  } catch (e) {
    notice(e.message);
  }
})();
