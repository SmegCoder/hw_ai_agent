from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .filesystem import open_read_file
from .skills import MAX_SKILL_BYTES, SKILL_NAME, Skill, SkillError
from .text import json_text


MAX_CATALOG_BYTES = 128 * 1024
DIGEST = re.compile(r"[0-9a-f]{64}")
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")


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
        data = json.loads(read_bytes(Path(file.anchor), file, limit))
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise MarketError(f"Невозможно прочитать JSON · {file.name}") from None
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
    source: Path
    check_on_start: bool
    auto_update: bool
    require_eval: bool


class SkillMarket:
    def __init__(self, config: Path, state_db: Path, workspace: Path):
        self.path = config.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()
        key = hashlib.sha256(os.fsencode(self.path)).hexdigest()[:24]
        self.db_path = state_db.expanduser().resolve().parent / f"skill-market-{key}.sqlite3"
        self.outside_workspace(self.path)
        self.outside_workspace(self.db_path)

    def outside_workspace(self, path: Path):
        if path.is_relative_to(self.workspace):
            raise MarketError("Конфигурация, каталог, пакеты и состояние маркета хранятся вне рабочей папки")

    def config(self) -> MarketConfig:
        raw = read_json(self.path, 8192)
        if (set(raw) - {"catalog", "check_on_start", "auto_update", "require_eval"} or not isinstance(raw.get("catalog"), str)
                or not raw["catalog"] or "\x00" in raw["catalog"]):
            raise MarketError("В конфигурации маркета требуется путь catalog")
        check, update = raw.get("check_on_start", True), raw.get("auto_update", False)
        gate = raw.get("require_eval", True)
        if any(type(value) is not bool for value in (check, update, gate)):
            raise MarketError("check_on_start, auto_update и require_eval должны быть true или false")
        source = (self.path.parent / raw["catalog"]).resolve()
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
            with closing(sqlite3.connect(self.db_path)) as db:
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
                """)
                with db:
                    yield db
        except (sqlite3.Error, OSError):
            raise MarketError("Невозможно открыть хранилище маркета скиллов") from None

    @staticmethod
    def record(db, event: str, *, source: Path | None, **data) -> dict:
        row = {"event": event, "time": datetime.now(timezone.utc).isoformat(),
               "source": str(source) if source else None, **data}
        db.execute("INSERT INTO events(data) VALUES (?)", (json_text(row),))
        return row

    @staticmethod
    def catalog(db, source: Path) -> dict[str, Offer]:
        saved = db.execute("SELECT entries FROM catalog WHERE source=?", (str(source),)).fetchone()
        if saved is None:
            return {}
        return {offer.name: offer for offer in map(Offer.parse, json.loads(saved["entries"]))}

    def read_package(self, source: Path, offer: Offer) -> bytes:
        target = source.parent / offer.path
        self.outside_workspace(target.resolve())
        try:
            content = read_bytes(source.parent, target, MAX_SKILL_BYTES)
            if hashlib.sha256(content).hexdigest() != offer.sha256:
                raise MarketError("SHA-256 пакета не совпадает с каталогом")
            Skill.parse(offer.name, content)
        except (OSError, UnicodeError, SkillError) as exc:
            raise MarketError(f"Некорректный пакет {offer.name} · {exc}") from None
        return content

    @staticmethod
    def activate(db, source: Path, offer: Offer, content: bytes):
        db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?)", (offer.name, offer.sha256, content))
        db.execute("INSERT OR REPLACE INTO active VALUES (?,?,?,?)",
                   (offer.name, str(source), offer.version, offer.sha256))

    def stage(self, db, source: Path, change: dict, content: bytes) -> dict | None:
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

    def check(self, *, startup: bool = False) -> tuple[Path | None, list[dict]]:
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            source = None
            try:
                config = self.config()
                source = config.source
                if startup and not config.check_on_start:
                    return source, []
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
            except MarketError as exc:
                return source, [self.record(db, "market_check", source=source, status="error", error=str(exc))]
            previous = self.catalog(db, source)
            events = [self.record(db, "market_check", source=source,
                                  status="unchanged" if offers == previous else "checked", count=len(offers))]
            db.execute("INSERT OR REPLACE INTO catalog VALUES (?,?)",
                       (str(source), json_text([asdict(offer) for offer in offers.values()])))
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
                    content = self.read_package(source, offer)
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

    def available(self, source: Path | None, excluded: set[str]):
        with self.database() as db:
            db.execute("BEGIN")
            active = dict(db.execute("SELECT name, sha256 FROM active"))
            skills = {name: self.revision(db, name, digest) for name, digest in active.items() if name not in excluded}
            offers = self.catalog(db, source) if source is not None else {}
            return skills, {name: offer for name, offer in offers.items() if name not in active and name not in excluded}

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

    def install(self, source: Path, offer: Offer, run_id: str) -> str:
        error = None
        with self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute("SELECT sha256 FROM installs WHERE run_id=? AND name=?",
                                 (run_id, offer.name)).fetchone()
            if receipt is not None and receipt["sha256"] == offer.sha256:
                return f"Installed {offer.name} {offer.version}. Call load_skill to load its instructions."
            current = db.execute("SELECT * FROM active WHERE name=?", (offer.name,)).fetchone()
            try:
                if self.config().source != source:
                    raise MarketError("Источник маркета изменился. Начните новый запуск")
                if current is None:
                    self.activate(db, source, offer, self.read_package(source, offer))
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
        with self.database() as db:
            db.execute("BEGIN")
            row = db.execute("""SELECT * FROM candidates WHERE name=? AND status='pending'
                                ORDER BY rowid DESC LIMIT 1""", (name,)).fetchone()
            if row is None:
                raise MarketError(f"Нет кандидата на проверку · {name}")
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
            source, name = Path(row["source"]), row["name"]
            current = db.execute("SELECT * FROM active WHERE name=?", (name,)).fetchone()
            if (config.source != source or not config.auto_update or not config.require_eval or current is None
                    or (current["source"], current["version"], current["sha256"]) !=
                    (str(source), row["old_version"], row["old_sha256"])):
                raise MarketError("Настройки или активная версия изменились во время сравнения")
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
            self.record(db, "market_rollback", source=Path(row["source"]), name=name, candidate_id=row["id"],
                        status="rolled_back", old_version=row["new_version"], old_sha256=row["new_sha256"],
                        new_version=row["old_version"], new_sha256=row["old_sha256"])
            return row["old_version"]

    def events(self):
        with self.database() as db:
            for row in db.execute("SELECT data FROM events ORDER BY seq"):
                yield row["data"]
