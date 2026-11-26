from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .filesystem import open_read_file


BUILTIN_SKILLS = Path(__file__).resolve().with_name("skill_cards")
MAX_SKILL_BYTES = 16_384
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    document: str
    digest: str

    @classmethod
    def read(cls, root: Path, folder: str) -> Skill:
        with open_read_file(root, root / folder / "SKILL.md") as descriptor:
            with os.fdopen(os.dup(descriptor), "rb") as source:
                data = source.read(MAX_SKILL_BYTES + 1)
        if len(data) > MAX_SKILL_BYTES:
            raise SkillError(f"SKILL.md exceeds {MAX_SKILL_BYTES} bytes")
        document = data.decode("utf-8")
        lines = document.splitlines()
        if not lines or lines[0] != "---" or "---" not in lines[1:]:
            raise SkillError("expected a header delimited by ---")
        end = lines.index("---", 1)
        try:
            header = yaml.compose("\n".join(lines[1:end]), Loader=yaml.BaseLoader)
        except (yaml.YAMLError, RecursionError):
            raise SkillError("invalid YAML in SKILL.md header") from None
        if not isinstance(header, yaml.MappingNode):
            raise SkillError("expected a YAML mapping with name and description")
        fields = {}
        for key, value in header.value:
            if not isinstance(key, yaml.ScalarNode) or key.value in fields:
                raise SkillError("expected distinct field names in the YAML header")
            fields[key.value] = value
        for key in ("name", "description"):
            if not isinstance(fields.get(key), yaml.ScalarNode) or fields[key].tag != "tag:yaml.org,2002:str":
                raise SkillError(f"expected a text {key}")
        if fields["name"].value != folder or len(folder) > 64 or not SKILL_NAME.fullmatch(folder):
            raise SkillError("name must match the folder and use lowercase letters, digits and hyphens")
        description = fields["description"].value.strip()
        if not 1 <= len(description) <= 1024 or not any(line.strip() for line in lines[end + 1:]):
            raise SkillError("expected a description of 1–1024 characters and a Markdown body")
        return cls(folder, description, document, hashlib.sha256(data).hexdigest())


class SkillCatalog:
    def __init__(self, workspace: Path, directory: Path | None = None):
        self.directory = directory.expanduser().resolve() if directory is not None else None
        workspace = workspace.resolve()
        if self.directory is not None:
            if self.directory.is_relative_to(workspace) or workspace.is_relative_to(self.directory):
                raise SkillError("--skills-dir must be outside the workspace")
        self.skills: dict[str, Skill] = {}
        self.errors: list[tuple[str, str]] = []
        for root in (BUILTIN_SKILLS, self.directory):
            if root is not None:
                self._discover(root)

    def _discover(self, root: Path) -> None:
        try:
            with os.scandir(root) as entries:
                folders = sorted(entry.name for entry in entries if entry.is_dir(follow_symlinks=False))
        except OSError as exc:
            self.errors.append((root.name, type(exc).__name__))
            return
        for folder in folders:
            if not os.path.lexists(root / folder / "SKILL.md"):
                continue
            try:
                skill = Skill.read(root, folder)
                if skill.name in self.skills:
                    raise SkillError("duplicate skill name")
                self.skills[skill.name] = skill
            except (OSError, UnicodeError, SkillError) as exc:
                reason = str(exc) if isinstance(exc, SkillError) else type(exc).__name__
                self.errors.append((folder, reason))

    def manifest(self) -> dict[str, str]:
        return {name: skill.digest for name, skill in self.skills.items()}

    def prompt(self) -> str:
        if not self.skills:
            return ""
        entries = [{"name": skill.name, "description": skill.description} for skill in self.skills.values()]
        return ("\nAvailable skills (name and description):\n" + json.dumps(entries, ensure_ascii=False)
                + "\nIf a skill clearly matches the request, call load_skill with its exact name before acting."
                  " Otherwise, proceed normally without loading a skill. Skill instructions cannot change runtime permissions.\n")

    def load(self, name: str, manifest: dict[str, str]) -> str:
        if name not in manifest:
            raise SkillError("skill name is not in the run's catalog")
        skill = self.skills.get(name)
        if skill is None or skill.digest != manifest[name]:
            raise SkillError("advertised SKILL.md changed or is missing")
        return skill.document


def skill_loaded_event(call: dict) -> dict:
    return {"event": "skill_loaded", "name": json.loads(call["function"]["arguments"])["name"],
            "tool_call_id": call["id"]}
