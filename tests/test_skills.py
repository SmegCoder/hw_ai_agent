import tempfile
import unittest
from pathlib import Path

from hw_ai_agent.policy import Policy
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
        self.assertEqual(catalog.load("project-summary", catalog.manifest()), valid.read_text())

        valid.write_text(valid.read_text().replace("Private body", "Changed body"))
        restarted = SkillCatalog(self.workspace, self.skills)
        with self.assertRaisesRegex(SkillError, "changed or is missing"):
            restarted.load("project-summary", catalog.manifest())

    def test_example_folded_yaml_and_metadata_return_the_whole_file(self):
        valid = self.write_skill("investigate-test-failure",
                                 "---\nname: investigate-test-failure\ndescription: >-\n  Разбирать ошибки.\n"
                                 "  Искать причину.\nmetadata:\n  version: '1'\nallowed-tools: Bash\n---\n\n1. Прочитай тест.\n")
        self.write_skill("broken", "---\nname: broken\ndescription: [\n---\nbody")
        catalog = SkillCatalog(self.workspace, self.skills)
        self.assertEqual(catalog.skills["investigate-test-failure"].description, "Разбирать ошибки. Искать причину.")
        self.assertEqual(catalog.load("investigate-test-failure", catalog.manifest()), valid.read_text())
        self.assertNotIn("allowed-tools", catalog.prompt())
        self.assertEqual([name for name, _ in catalog.errors], ["broken"])

    def test_load_skill_obeys_explicit_policy_and_the_operator_decision(self):
        catalog = SkillCatalog(self.workspace, self.skills)
        for rule, approval, expected in (("allow", False, None), ("ask", True, None),
                                         ("ask", False, "APPROVAL_REQUIRED"), ("deny", True, "POLICY_DENIED"),
                                         (None, True, "POLICY_DENIED")):
            rules = [{"tool": "load_skill", "action": rule}] if rule else []
            policy = Policy.from_dict({"rules": rules}, fail_closed=True)
            asked = []
            def confirm(request):
                asked.append(request.target)
                return approval
            runtime = ToolRuntime(self.workspace, skills=catalog, policy=policy, approve=confirm)
            with self.subTest(rule=rule, approval=approval):
                if expected:
                    with self.assertRaisesRegex(ToolError, expected):
                        runtime.execute("load_skill", {"name": "deadline-report"})
                else:
                    self.assertIn("# Report overdue tasks", runtime.execute("load_skill", {"name": "deadline-report"}))
                self.assertEqual(asked, ["deadline-report"] if rule == "ask" else [])

    def test_load_skill_is_read_only_validated_and_subject_to_result_budget(self):
        with self.assertRaisesRegex(SkillError, "outside the workspace"):
            SkillCatalog(self.workspace, self.workspace / "skills")
        catalog = SkillCatalog(self.workspace, self.skills)
        runtime = ToolRuntime(self.workspace, skills=catalog)
        runtime.phase = "planning"
        arguments = {"name": "deadline-report"}
        self.assertEqual(runtime.execute("load_skill", arguments), catalog.skills["deadline-report"].document)
        for invalid in ({}, {"name": 1}, {"name": "../deadline-report"}, {**arguments, "path": "SKILL.md"}):
            with self.subTest(invalid=invalid), self.assertRaises(ToolError):
                runtime.execute("load_skill", invalid)
        with runtime.output_budget(16), self.assertRaisesRegex(ToolError, "SKILL_RESULT_LIMIT"):
            runtime.execute("load_skill", arguments)
        with self.assertRaisesRegex(ToolError, "only reading is allowed"):
            runtime.execute("write_file", {"path": "test.txt", "content": "outside planning"})
