from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .anthropic_market import ANTHROPIC_REPOSITORY, PublicMarketError, read_publication
from .filesystem import open_read_file
from .skills import MAX_SKILL_BYTES, SKILL_NAME, Skill, SkillError
from .text import json_text


MAX_CATALOG_BYTES = 128 * 1024
DIGEST = re.compile(r"[0-9a-f]{64}")
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")
LEGACY_DATABASE = re.compile(r"skill-market-[0-9a-f]{24}\.sqlite3")
ANTHROPIC_MARKET = "anthropic"
MarketSource = Path | str


class MarketError(ValueError):
    pass


def read_bytes(root: Path, file: Path, limit: int) -> bytes:
    with open_read_file(root, file) as descriptor, os.fdopen(os.dup(descriptor), "rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise MarketError(f"{file.name} exceeds the market size limit")
    return content


def read_json(file: Path, limit: int) -> dict:
    try:
        file.stat()
        raw = read_bytes(Path(file.anchor), file, limit)
    except FileNotFoundError:
        raise MarketError(f"Файл JSON не найден · {file.name}") from None
    except OSError:
        raise MarketError(f"Не удалось открыть файл JSON · {file.name}") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MarketError(f"Некорректный JSON · {file.name}, строка {exc.lineno}, столбец {exc.colno}") from None
    except (ValueError, RecursionError):
        raise MarketError(f"Некорректный JSON · {file.name}") from None
    if not isinstance(data, dict):
        raise MarketError(f"Ожидается объект JSON · {file.name}")
    return data


@dataclass(frozen=True)
class Offer:
    name: str
    description: str
    version: str
    path: str
    sha256: str

    @classmethod
    def parse(cls, value: dict) -> Offer:
        fields = {"name", "description", "version", "path", "sha256"}
        if not isinstance(value, dict) or set(value) != fields or any(type(v) is not str for v in value.values()):
            raise MarketError("Ожидаются поля name, description, version, path и sha256")
        offer = cls(**value)
        path = Path(offer.path)
        if (not SKILL_NAME.fullmatch(offer.name) or len(offer.name) > 64
                or not offer.description.strip() or len(offer.description) > 1024
                or not VERSION.fullmatch(offer.version) or not DIGEST.fullmatch(offer.sha256)
                or path.is_absolute() or ".." in path.parts or path.name != "SKILL.md"
                or "\x00" in offer.path or len(offer.path) > 512):
            raise MarketError("Некорректное описание пакета в каталоге")
        return offer


@dataclass(frozen=True)
class MarketConfig:
    source: MarketSource
    check_on_start: bool
    auto_update: bool
    require_eval: bool


@dataclass(frozen=True)
class CatalogSnapshot:
    offers: dict[str, Offer]
    cards: dict[str, Skill] = field(default_factory=dict)
    audit: dict = field(default_factory=dict)


class SkillMarket:
    def __init__(self, config: Path | str | None, state_db: Path, workspace: Path):
        self.path = config.expanduser().resolve() if isinstance(config, Path) else None
        self.reference = str(self.path) if self.path is not None else config
        self.workspace = workspace.expanduser().resolve()
        self.db_path = state_db.expanduser().resolve().parent / "skill-market.sqlite3"
        if self.path is not None:
            self.outside_workspace(self.path)
        self.outside_workspace(self.db_path)

    def outside_workspace(self, path: Path):
        if path.is_relative_to(self.workspace):
            raise MarketError("Конфигурация, каталог, пакеты и состояние маркета хранятся вне рабочей папки")

    def config(self) -> MarketConfig:
        if self.reference == ANTHROPIC_MARKET:
            return MarketConfig(ANTHROPIC_REPOSITORY, check_on_start=True, auto_update=True, require_eval=True)
        if self.path is None:
            raise MarketError("Укажите --market-config для работы с источником маркета")
        raw = read_json(self.path, 8192)
        if (set(raw) - {"catalog", "marketplace", "check_on_start", "auto_update", "require_eval"}
                or len(raw.keys() & {"catalog", "marketplace"}) != 1):
            raise MarketError('Укажите один источник маркета: путь catalog или marketplace "anthropic"')
        check, update = raw.get("check_on_start", True), raw.get("auto_update", False)
        gate = raw.get("require_eval", True)
        if any(type(value) is not bool for value in (check, update, gate)):
            raise MarketError("check_on_start, auto_update и require_eval должны быть true или false")
        if "marketplace" in raw:
            if raw["marketplace"] != ANTHROPIC_MARKET:
                raise MarketError('Поддерживаемый публичный marketplace: "anthropic"')
            return MarketConfig(ANTHROPIC_REPOSITORY, check, update, gate)
        catalog = raw["catalog"]
        if not isinstance(catalog, str) or not catalog or "\x00" in catalog:
            raise MarketError("В конфигурации маркета требуется путь catalog")
        source = (self.path.parent / catalog).resolve()
        self.outside_workspace(source)
        return MarketConfig(source, check, update, gate)

    @contextmanager
    def database(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.db_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise MarketError("Хранилище маркета должно быть обычным файлом")
            finally:
                os.close(descriptor)
            with closing(sqlite3.connect(self.db_path, timeout=35)) as db:
                db.row_factory = sqlite3.Row
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS catalog (source TEXT PRIMARY KEY, entries TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS active (name TEXT PRIMARY KEY, source TEXT NOT NULL,
                        version TEXT NOT NULL, sha256 TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS versions (name TEXT NOT NULL, sha256 TEXT NOT NULL,
                        body BLOB NOT NULL, PRIMARY KEY(name, sha256));
                    CREATE TABLE IF NOT EXISTS installs (run_id TEXT NOT NULL, name TEXT NOT NULL,
                        sha256 TEXT NOT NULL, PRIMARY KEY(run_id, name));
                    CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                        source TEXT NOT NULL, old_version TEXT NOT NULL, old_sha256 TEXT NOT NULL,
                        new_version TEXT NOT NULL, new_sha256 TEXT NOT NULL, status TEXT NOT NULL, decision TEXT);
                    CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, data TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS legacy_imports (name TEXT PRIMARY KEY);
                """)
                self.import_legacy(db)
                with db:
                    yield db
        except (sqlite3.Error, OSError):
            raise MarketError("Невозможно открыть хранилище маркета скиллов") from None

    def import_legacy(self, db):
        imported = {row["name"] for row in db.execute("SELECT name FROM legacy_imports")}
        paths = sorted(path for path in self.db_path.parent.iterdir()
                       if LEGACY_DATABASE.fullmatch(path.name) and path.name not in imported)
        if not paths:
            return
        with db:
            db.execute("BEGIN IMMEDIATE")
            for path in paths:
                if not db.execute("INSERT OR IGNORE INTO legacy_imports VALUES (?)", (path.name,)).rowcount:
                    continue
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise MarketError("Прежнее хранилище маркета должно быть обычным файлом")
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=35)) as legacy:
                    legacy.execute("BEGIN")
                    for table, placeholders in (("catalog", "?,?"), ("versions", "?,?,?"),
                                                ("active", "?,?,?,?"), ("installs", "?,?,?"),
                                                ("candidates", "?,?,?,?,?,?,?,?,?")):
                        db.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})",
                                       legacy.execute(f"SELECT * FROM {table}"))
                    db.executemany("INSERT INTO events(data) VALUES (?)",
                                   legacy.execute("SELECT data FROM events ORDER BY seq"))

    @staticmethod
    def record(db, event: str, *, source: MarketSource | None, **data) -> dict:
        row = {"event": event, "time": datetime.now(timezone.utc).isoformat(),
               "source": str(source) if source else None, **data}
        db.execute("INSERT INTO events(data) VALUES (?)", (json_text(row),))
        return row

    @staticmethod
    def catalog(db, source: MarketSource) -> dict[str, Offer]:
        saved = db.execute("SELECT entries FROM catalog WHERE source=?", (str(source),)).fetchone()
        if saved is None:
            return {}
        return {offer.name: offer for offer in map(Offer.parse, json.loads(saved["entries"]))}

    @staticmethod
    def read_catalog(source: MarketSource) -> CatalogSnapshot:
        if isinstance(source, str):
            try:
                publication = read_publication()
            except PublicMarketError as exc:
                raise MarketError(str(exc)) from None
            offers = {name: Offer(name, card.description, card.digest[:12], f"skills/{name}/SKILL.md", card.digest)
                      for name, card in publication.cards.items()}
            return CatalogSnapshot(offers, publication.cards, {"revision": publication.revision,
                                                              "skipped": publication.skipped})
        data = read_json(source, MAX_CATALOG_BYTES)
        entries = data.get("skills")
        if set(data) != {"skills"} or not isinstance(entries, list) or len(entries) > 100:
            raise MarketError("Каталог должен содержать список skills из максимум 100 записей")
        offers = {}
        for item in entries:
            offer = Offer.parse(item)
            if offer.name in offers:
                raise MarketError(f"Дубликат в каталоге · {offer.name}")
            offers[offer.name] = offer
        return CatalogSnapshot(offers)

    def read_package(self, db, source: MarketSource, offer: Offer) -> bytes:
        try:
            if isinstance(source, str):
                return self.revision(db, offer.name, offer.sha256).document.encode("utf-8")
            target = source.parent / offer.path
            self.outside_workspace(target.resolve())
            content = read_bytes(source.parent, target, MAX_SKILL_BYTES)
            if hashlib.sha256(content).hexdigest() != offer.sha256:
                raise MarketError("SHA-256 пакета не совпадает с каталогом")
            Skill.parse(offer.name, content)
        except (OSError, UnicodeError, SkillError) as exc:
            raise MarketError(f"Некорректный пакет {offer.name} · {exc}") from None
        return content

    @staticmethod
    def activate(db, source: MarketSource, offer: Offer, content: bytes):
        db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?)", (offer.name, offer.sha256, content))
        db.execute("INSERT OR REPLACE INTO active VALUES (?,?,?,?)",
                   (offer.name, str(source), offer.version, offer.sha256))

    def stage(self, db, source: MarketSource, change: dict, content: bytes) -> dict | None:
        identifier = hashlib.sha256(json_text([str(source), change], sort_keys=True).encode()).hexdigest()[:32]
        if db.execute("SELECT 1 FROM candidates WHERE id=?", (identifier,)).fetchone():
            return None
        db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?)",
                   (change["name"], change["new_sha256"], content))
        db.execute("""UPDATE candidates SET status='superseded'
                      WHERE name=? AND source=? AND status='pending'""", (change["name"], str(source)))
        db.execute("""INSERT INTO candidates VALUES (?,?,?,?,?,?,?,'pending',NULL)""",
                   (identifier, change["name"], str(source), change["old_version"], change["old_sha256"],
                    change["new_version"], change["new_sha256"]))
        return self.record(db, "market_update", source=source, status="pending", candidate_id=identifier, **change)

    def check(self, *, startup: bool = False) -> tuple[MarketSource | None, list[dict]]:
        if startup and self.reference is None:
            return None, []
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            source = None
            try:
                config = self.config()
                source = config.source
                if startup and not config.check_on_start:
                    return source, []
                if startup and self.reference == ANTHROPIC_MARKET and self.catalog(db, source):
                    return source, []
                snapshot = self.read_catalog(source)
            except MarketError as exc:
                return source, [self.record(db, "market_check", source=source, status="error", error=str(exc))]
            offers = snapshot.offers
            previous = self.catalog(db, source)
            events = [self.record(db, "market_check", source=source,
                                  status="unchanged" if offers == previous else "checked", count=len(offers),
                                  **snapshot.audit)]
            db.execute("INSERT OR REPLACE INTO catalog VALUES (?,?)",
                       (str(source), json_text([asdict(offer) for offer in offers.values()])))
            db.executemany("INSERT OR IGNORE INTO versions VALUES (?,?,?)",
                           ((name, card.digest, card.document.encode("utf-8"))
                            for name, card in snapshot.cards.items()))
            installed = {row["name"]: row for row in db.execute("SELECT * FROM active")}
            for name, offer in offers.items():
                old = installed.get(name)
                if old is None:
                    if name not in previous:
                        events.append(self.record(db, "market_new", source=source, name=name,
                                                  new_version=offer.version, new_sha256=offer.sha256))
                    continue
                if old["source"] != str(source) or (old["version"], old["sha256"]) == (offer.version, offer.sha256):
                    continue
                change = {"name": name, "old_version": old["version"], "new_version": offer.version,
                          "old_sha256": old["sha256"], "new_sha256": offer.sha256}
                if not config.auto_update:
                    events.append(self.record(db, "market_update", source=source, status="available", **change))
                    continue
                try:
                    content = self.read_package(db, source, offer)
                except MarketError as exc:
                    events.append(self.record(db, "market_update", source=source, status="error",
                                              error=str(exc), **change))
                else:
                    if config.require_eval:
                        event = self.stage(db, source, change, content)
                        if event is not None:
                            events.append(event)
                    else:
                        self.activate(db, source, offer, content)
                        events.append(self.record(db, "market_update", source=source, status="updated", **change))
            return source, events

    @staticmethod
    def revision(db, name: str, digest: str) -> Skill:
        row = db.execute("SELECT body FROM versions WHERE name=? AND sha256=?", (name, digest)).fetchone()
        if row is None:
            raise MarketError(f"Сохранённая версия скилла недоступна · {name}")
        skill = Skill.parse(name, row["body"])
        if skill.digest != digest:
            raise MarketError(f"Повреждена сохранённая версия скилла · {name}")
        return skill

    def available(self, source: MarketSource | None, excluded: set[str]):
        with self.database() as db:
            db.execute("BEGIN")
            active = dict(db.execute("SELECT name, sha256 FROM active"))
            skills = {name: self.revision(db, name, digest) for name, digest in active.items() if name not in excluded}
            offers = self.catalog(db, source) if source is not None else {}
            return skills, {name: offer for name, offer in offers.items() if name not in active and name not in excluded}

    def search(self, source: MarketSource, query: str) -> list[Offer]:
        terms = query.casefold().split()
        with self.database() as db:
            offers = self.catalog(db, source)
        return [offer for _, offer in sorted(offers.items())
                if all(term in f"{offer.name} {offer.description}".casefold() for term in terms)]

    def pinned(self, manifest: dict) -> dict[str, Skill]:
        with self.database() as db:
            db.execute("BEGIN")
            return {name: self.revision(db, name, digest) for name, digest in manifest.items()}

    def installed_by_run(self, run_id: str, offer: Offer) -> Skill:
        with self.database() as db:
            receipt = db.execute("SELECT sha256 FROM installs WHERE run_id=? AND name=?",
                                 (run_id, offer.name)).fetchone()
            if receipt is None or receipt["sha256"] != offer.sha256:
                raise MarketError("Сначала установите скилл через install_skill")
            return self.revision(db, offer.name, offer.sha256)

    def install(self, source: MarketSource, offer: Offer, run_id: str) -> str:
        error = None
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute("SELECT sha256 FROM installs WHERE run_id=? AND name=?",
                                 (run_id, offer.name)).fetchone()
            if receipt is not None and receipt["sha256"] == offer.sha256:
                return f"Installed {offer.name} {offer.version}. Call load_skill to load its instructions."
            current = db.execute("SELECT * FROM active WHERE name=?", (offer.name,)).fetchone()
            try:
                configured_source = self.config().source
                if str(configured_source) != str(source):
                    raise MarketError("Источник маркета изменился. Начните новый запуск")
                if current is None:
                    self.activate(db, source, offer, self.read_package(db, configured_source, offer))
                elif (current["source"], current["version"], current["sha256"]) != (
                        str(source), offer.version, offer.sha256):
                    raise MarketError("Уже установлена другая версия скилла. Начните новый запуск")
                db.execute("INSERT INTO installs VALUES (?,?,?)", (run_id, offer.name, offer.sha256))
            except MarketError as exc:
                error = str(exc)
            self.record(db, "market_install", source=source, name=offer.name, run_id=run_id,
                        status="error" if error else "already_installed" if current else "installed",
                        old_version=current["version"] if current else None, new_version=offer.version,
                        old_sha256=current["sha256"] if current else None, new_sha256=offer.sha256, error=error)
        if error:
            raise MarketError(error)
        return f"Installed {offer.name} {offer.version}. Call load_skill to load its instructions."

    def installed(self) -> list[dict]:
        with self.database() as db:
            return [dict(row) for row in db.execute("SELECT * FROM active ORDER BY name")]

    def candidates(self) -> list[dict]:
        with self.database() as db:
            return [{**row, "decision": json.loads(row["decision"]) if row["decision"] else None}
                    for row in db.execute("SELECT * FROM candidates ORDER BY rowid")]

    def candidate(self, name: str) -> tuple[dict, Skill, Skill]:
        config = self.config()
        if not config.auto_update or not config.require_eval:
            raise MarketError("Для eval кандидата включите auto_update и require_eval")
        with self.database() as db:
            db.execute("BEGIN")
            row = db.execute("""SELECT c.* FROM candidates c JOIN active a ON c.name=a.name
                                AND c.source=a.source AND c.old_version=a.version AND c.old_sha256=a.sha256
                                WHERE c.name=? AND c.source=? AND c.status='pending'
                                ORDER BY c.rowid DESC LIMIT 1""", (name, str(config.source))).fetchone()
            if row is None:
                raise MarketError(f"Нет кандидата на проверку · {name}")
            published = self.read_catalog(config.source).offers.get(name)
            if published is None or (published.version, published.sha256) != (row["new_version"], row["new_sha256"]):
                raise MarketError("Кандидат больше не опубликован в текущем каталоге")
            return dict(row), self.revision(db, name, row["old_sha256"]), self.revision(db, name, row["new_sha256"])

    def finish_candidate(self, identifier: str, decision: dict) -> dict:
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM candidates WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise MarketError("Кандидат не найден")
            if row["decision"]:
                return json.loads(row["decision"])
            if row["status"] != "pending":
                raise MarketError("Кандидат заменён новой версией. Проверьте актуальный каталог")
            config = self.config()
            source, name = config.source, row["name"]
            current = db.execute("SELECT * FROM active WHERE name=?", (name,)).fetchone()
            if (str(source) != row["source"] or not config.auto_update or not config.require_eval or current is None
                    or (current["source"], current["version"], current["sha256"]) !=
                    (str(source), row["old_version"], row["old_sha256"])):
                raise MarketError("Настройки или активная версия изменились во время сравнения")
            published = self.read_catalog(source).offers.get(name)
            if published is None or (published.version, published.sha256) != (row["new_version"], row["new_sha256"]):
                raise MarketError("Опубликованный кандидат изменился во время сравнения")
            status = decision["status"]
            if status not in {"accepted", "rejected"}:
                raise MarketError("Неверное решение по кандидату")
            if status == "accepted":
                db.execute("UPDATE active SET version=?, sha256=? WHERE name=?",
                           (row["new_version"], row["new_sha256"], name))
            db.execute("UPDATE candidates SET status=?, decision=? WHERE id=?",
                       (status, json_text(decision), identifier))
            self.record(db, "market_decision", source=source, name=name, status=status, candidate_id=identifier,
                        old_version=row["old_version"], old_sha256=row["old_sha256"], new_version=row["new_version"],
                        new_sha256=row["new_sha256"], reasons=decision["reasons"], report=decision["report"])
            return decision

    def rollback(self, name: str) -> str:
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT c.* FROM candidates c JOIN active a ON c.name=a.name
                                AND c.source=a.source AND c.new_version=a.version AND c.new_sha256=a.sha256
                                WHERE c.name=? AND c.status='accepted' ORDER BY c.rowid DESC LIMIT 1""",
                             (name,)).fetchone()
            if row is None:
                raise MarketError(f"Нет принятого обновления для отката · {name}")
            self.revision(db, name, row["old_sha256"])
            db.execute("UPDATE active SET version=?, sha256=? WHERE name=?",
                       (row["old_version"], row["old_sha256"], name))
            db.execute("UPDATE candidates SET status='rolled_back' WHERE id=?", (row["id"],))
            self.record(db, "market_rollback", source=row["source"], name=name, candidate_id=row["id"],
                        status="rolled_back", old_version=row["new_version"], old_sha256=row["new_sha256"],
                        new_version=row["old_version"], new_sha256=row["old_sha256"])
            return row["old_version"]

    def recent_events(self, limit: int) -> list[dict]:
        with self.database() as db:
            rows = db.execute("SELECT data FROM events ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
            return [json.loads(row["data"]) for row in reversed(rows)]

    def events(self):
        with self.database() as db:
            for row in db.execute("SELECT data FROM events ORDER BY seq"):
                yield row["data"]
