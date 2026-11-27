"""Read-only workspace browser for a saved dialogue."""
import json
import os
import stat
from pathlib import Path

from . import cli
from .filesystem import open_directory, open_read_file
from .journal import ChangeJournal
from .policy import Action, Policy
from .text import has_surrogates
from .tools import protected


PREVIEW_BYTES = 96 * 1024


class ChatFiles:
    def __init__(self, store, identifier):
        store.validate_id(identifier)
        row = store.db.execute("SELECT configuration FROM conversations WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise ValueError("Чат не найден")
        binding = json.loads(row["configuration"])["binding"]
        self.root = Path(binding["workspace"])
        if not self.root.is_absolute() or self.root.resolve() != self.root or store.path.is_relative_to(self.root):
            raise ValueError("Папка чата недоступна")
        self.policy = Policy.restore(binding["policy"], builtin=binding.get("policy_source") == "default")
        if binding.get("policy_path"):
            current = cli.load_policy(Path(binding["policy_path"]), self.root)
            if current.as_dict() != self.policy.as_dict():
                raise ValueError("Права доступа изменились с момента создания чата")
        self.store = store
        self.id = identifier

    def allowed(self, path, directory=False):
        if (path.is_absolute() or ".." in path.parts or protected(path) or has_surrogates(str(path))):
            return False
        ancestors = list(path.parents)
        if directory:
            ancestors.append(path)
        for parent in ancestors:
            if self.policy.decide("list_files", path=str(parent)).action != Action.ALLOW:
                return False
            if any(parent.is_relative_to(d) for d in self.policy.directory_deny):
                return False
        if directory:
            return self.policy.directory_allowed(str(path)) or any(
                Path(d).is_relative_to(path) for d in self.policy.directory_allow)
        return (self.policy.directory_allowed(str(path))
                and self.policy.decide("read_file", path=str(path)).action == Action.ALLOW)

    def listing(self, directory):
        relative = Path(directory)
        if not self.allowed(relative, directory=True):
            raise ValueError("Нет доступа к каталогу")
        items = []
        with open_directory(Path(self.root.anchor), self.root / relative) as fd, os.scandir(fd) as entries:
            for item in entries:
                path = relative / item.name
                try:
                    mode = item.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                folder = stat.S_ISDIR(mode.st_mode)
                if not folder and not (stat.S_ISREG(mode.st_mode) and mode.st_nlink == 1):
                    continue
                if self.allowed(path, folder):
                    items.append({"name": item.name, "directory": folder})
        rows = self.store.db.execute("""
            SELECT e.path, t.number, t.session_id, j.status FROM journal_entries e
            JOIN conversation_turns t USING(session_id) JOIN journals j USING(session_id)
            WHERE t.conversation_id=? AND e.before_state != e.after_state
            ORDER BY t.number DESC, e.path""", (self.id,))
        changes = [{"path": r["path"], "turn": r["number"], "run": r["session_id"],
                    "rolled_back": r["status"] == "rolled_back"} for r in rows
                   if isinstance(r["path"], str) and self.allowed(Path(r["path"]))]
        return {"items": sorted(items, key=lambda i: (not i["directory"], i["name"].casefold())),
                "changes": changes}

    def preview(self, name, run=None):
        path = Path(name)
        if path == Path(".") or not self.allowed(path):
            raise ValueError("Нет доступа к файлу")
        if run is not None:
            row = self.store.db.execute("SELECT 1 FROM conversation_turns WHERE session_id=? AND conversation_id=?",
                                        (run, self.id)).fetchone()
            if not row:
                raise ValueError("Запрос не принадлежит чату")
            journal = ChangeJournal(self.store, run, None)
            entry = journal.entries().get(str(path))
            if not entry:
                raise ValueError("Для файла нет сохранённого изменения")
            for state in entry.values():
                if state and (state["kind"] != "file" or state["size"] > PREVIEW_BYTES):
                    raise ValueError("Сравнение доступно для текстовых файлов до 96 КиБ")
                if state:
                    self.decode(journal.blob(state))
            return {"text": journal.diff({str(path): entry})}
        with open_read_file(Path(self.root.anchor), self.root / path) as fd, os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(PREVIEW_BYTES + 1)
        if len(data) > PREVIEW_BYTES:
            raise ValueError("Предпросмотр доступен для файлов до 96 КиБ")
        return {"text": self.decode(data)}

    @staticmethod
    def decode(data):
        if b"\0" not in data:
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                pass
        raise ValueError("Для двоичного файла или этой кодировки нет текстового предпросмотра")
