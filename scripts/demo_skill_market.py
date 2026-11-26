"""Exercise the market with real runtime operations and scripted model responses."""

import json
import shutil
import tempfile
from pathlib import Path

from hw_ai_agent.agent import CodingAgent
from hw_ai_agent.session import SessionStore
from hw_ai_agent.skill_market import SkillMarket
from hw_ai_agent.skills import SkillCatalog
from hw_ai_agent.tools import ToolRuntime


MARKET_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "skill-market"
WEEK = "study-week-overview"
REQUEST = "Составь план учёбы на неделю"


class DemoModel:
    def complete(self, messages, tools):
        schemas = {tool["function"]["name"] for tool in tools}
        loaded = any(msg.get("name") == "load_skill" for msg in messages)
        selected = any(msg.get("name") == "install_skill" for msg in messages)
        if loaded:
            return {"content": "Скилл загружен."}
        if WEEK not in messages[0]["content"] or "load_skill" not in schemas:
            raise RuntimeError("Expected the weekly planning skill in the catalog")
        offer = messages[0]["content"].partition("Skill market offers")[2]
        tool = "install_skill" if not selected and WEEK in offer else "load_skill"
        if tool not in schemas:
            raise RuntimeError(f"Expected tool {tool}")
        return {"tool_calls": [{"id": tool, "type": "function",
                                "function": {"name": tool, "arguments": json.dumps({"name": WEEK})}}]}


def main():
    with tempfile.TemporaryDirectory(prefix="agi-market-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        workspace.mkdir()
        source = root / "registry"
        shutil.copytree(MARKET_EXAMPLE, source)
        config = source / "market.json"
        database = root / "state" / "sessions.sqlite3"
        market = SkillMarket(config, database, workspace)

        def publish(version):
            shutil.copyfile(source / f"catalog-{version}.json", source / "catalog.json")

        def run():
            runtime = ToolRuntime(workspace, skills=SkillCatalog(workspace), market=market,
                                  approve=lambda request: request.tool == "install_skill")
            with SessionStore(database) as store:
                agent = CodingAgent(DemoModel(), runtime, store=store)
                agent.run(REQUEST)
                state = store.get(agent.session_id)
                [loaded] = [message["content"] for message in state.messages if message.get("name") == "load_skill"]
                if not loaded.startswith("---\nname:"):
                    raise RuntimeError("The ordinary loader did not return the installed skill")

        publish("v1")
        run()
        publish("v2")
        config.write_text(json.dumps({"catalog": "catalog.json", "check_on_start": True, "auto_update": False}))
        run()
        if market.installed()[0]["version"] != "1.0.0":
            raise RuntimeError("Disabled auto-update replaced the active version")

        shutil.copyfile(MARKET_EXAMPLE / "market.json", config)
        run()
        run()
        publish("invalid")
        run()
        if [(item["name"], item["version"]) for item in market.installed()] != [(WEEK, "1.1.0")]:
            raise RuntimeError("Unexpected installed skills after the malformed update")
        for line in market.events():
            event = json.loads(line)
            if event["source"] is not None:
                event["source"] = Path(event["source"]).relative_to(root.resolve()).as_posix()
            print(json.dumps(event, ensure_ascii=False))


if __name__ == "__main__":
    main()
