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
async function listSessions() {
  sessions = await api("sessions");
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
function renderChat() {
  const turns = chat?.turns || [];
  const atBottom =
      $("conversation").scrollHeight - $("conversation").scrollTop - $("conversation").clientHeight < 45;
  $("title").textContent = turns.length ? turns[0].prompt.slice(0, 75) : "Новый диалог";
  $("model").textContent = chat?.model || "";
  const pending = chat?.approval;
  $("messages").innerHTML = turns.length ?
      turns
          .map(
              t => "<div class=\"turn user\"><b>Вы</b><p>" + esc(t.prompt) + "</p></div>" +
                  (t.answer ? "<div class=\"turn agent\"><b>agi</b><p>" + esc(t.answer) + "</p></div>" : "") +
                  detail("Статистика запроса · " + t.status, metrics(t.stats)))
          .join("") :
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
                                      pending       ? "Ожидает решения" :
                                      chat?.running ? "Агент работает…" :
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
    chat = result;
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
  $("secondary").hidden = page === "chat";
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
      card("Модель",
           select(
               "provider", "Провайдер", Object.keys({ollama: 1, openrouter: 1, openai: 1}).map(x => [x, x]),
               p.provider) +
               input("modelName", "ID модели", p.model, "Из .env по умолчанию") +
               info("Ключ берётся из окружения сервера.")) +
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
               info("Автообновление создаёт кандидата. До применения сравниваются обе версии.")) +
      "</div><div class=\"buttons\">" + button("save", "Сохранить настройки", "", "primary") +
      button("check", "Проверить сохранённый источник") +
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
      id = await api("new", {});
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
    id = action === "session" ? v : "";
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
  } else if (action === "save") {
    const disabled = new Set(prefs.disabled);
    document.querySelectorAll("[data-skill]")
        .forEach(el => el.checked ? disabled.delete(el.dataset.skill) : disabled.add(el.dataset.skill));
    prefs = await api("settings", {
      provider: $("provider").value,
      model: $("modelName").value,
      mcp: $("mcpPath").value,
      source: $("source").value,
      demo: $("demo").checked,
      skills_dir: $("skillsDir").value,
      check_on_start: $("checkStart").checked,
      auto_update: $("autoUpdate").checked,
      disabled: [...disabled]
    });
    renderSettings();
    notice("Настройки сохранены.");
  } else if (action === "mcp") {
    const data = await api("mcp");
    $("mcpResult").innerHTML =
        data.tools
                .map(t => detail(t.function.name, esc(t.function.description) + json(t.function.parameters)))
                .join("") +
            data.events.filter(e => e.error).map(e => info(e.error)).join("") ||
        info("Нет подключённых инструментов. Сохрани путь к конфигурации.");
  } else if (action === "check" || action === "install") {
    await api(action, {name: v});
    await show("settings");
    notice(action === "check" ? "Каталог проверен." : "Скилл установлен.");
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
    await listSessions();
    const saved = localStorage.getItem("agi.chat");
    if (sessions.some(s => s.id === saved)) id = saved;
    await listSessions();
    await show("chat");
    setInterval(refresh, 1100);
  } catch (e) {
    notice(e.message);
  }
})();
