import tempfile
import unittest
from pathlib import Path

from hw_ai_agent.skills import SkillCatalog, SkillError
from hw_ai_agent.tools import ToolError, ToolRuntime


class SkillCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.skills = root / "skills"
        self.skills.mkdir()

    def write_skill(self, name, content):
        folder = self.skills / name
        folder.mkdir(exist_ok=True)
        document = folder / "SKILL.md"
        document.write_text(content, encoding="utf-8")
        return document

    def test_rediscovers_valid_files_and_skips_bad_metadata_duplicates_and_links(self):
        initial = SkillCatalog(self.workspace, self.skills)
        self.write_skill("invalid", "missing metadata")
        valid = self.write_skill("project-summary", '---\nname: project-summary\n'
                                 'description: "Summarize a project."\n---\nPrivate body\n')
        self.write_skill("deadline-report", "---\nname: deadline-report\ndescription: Duplicate\n---\nReplacement\n")
        link = self.skills / "link"
        link.mkdir()
        (link / "SKILL.md").symlink_to(valid)

        catalog = SkillCatalog(self.workspace, self.skills)
        self.assertNotIn("project-summary", initial.manifest())
        self.assertIn("project-summary", catalog.manifest())
        self.assertNotIn("Private body", catalog.prompt())
        self.assertNotIn("Duplicate", catalog.prompt())
        self.assertEqual({name for name, _ in catalog.errors}, {"invalid", "deadline-report", "link"})
        self.assertEqual(catalog.load("project-summary", catalog.manifest()), "Private body")

        valid.write_text(valid.read_text().replace("Private body", "Changed body"))
        restarted = SkillCatalog(self.workspace, self.skills)
        with self.assertRaisesRegex(SkillError, "changed or is missing"):
            restarted.load("project-summary", catalog.manifest())

    def test_load_skill_is_read_only_validated_and_subject_to_result_budget(self):
        with self.assertRaisesRegex(SkillError, "outside the workspace"):
            SkillCatalog(self.workspace, self.workspace / "skills")
        catalog = SkillCatalog(self.workspace, self.skills)
        runtime = ToolRuntime(self.workspace, skills=catalog)
        runtime.phase = "planning"
        arguments = {"name": "deadline-report"}
        self.assertTrue(runtime.execute("load_skill", arguments).startswith("# Report overdue tasks"))
        for invalid in ({}, {"name": 1}, {"name": "../deadline-report"}, {**arguments, "path": "SKILL.md"}):
            with self.subTest(invalid=invalid), self.assertRaises(ToolError):
                runtime.execute("load_skill", invalid)
        with runtime.output_budget(16), self.assertRaisesRegex(ToolError, "SKILL_RESULT_LIMIT"):
            runtime.execute("load_skill", arguments)
        with self.assertRaisesRegex(ToolError, "only reading is allowed"):
            runtime.execute("write_file", {"path": "test.txt", "content": "outside planning"})
