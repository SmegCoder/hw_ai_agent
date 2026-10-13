from __future__ import annotations

import copy
import json
from .policy import ApprovalDeferred, TOOLS


class PlanError(ValueError):
    pass


class PlanApprovalDeferred(ApprovalDeferred):
    pass


PLAN_INSTRUCTIONS = '''
This run begins in a planning phase. Inspect the workspace using only list_files
and read_file. Do not call Bash, write_file or delete_file before plan approval.
Submit your structured plan with submit_plan, in Russian. Include the goal,
ordered steps and their tools, risks, assumptions and verifiable done_criteria.
The runtime will request an explicit operator decision. Only after it reports
approval may you execute the plan using the normal tools and permission rules.
A rejected plan must not be executed. Tool errors are recoverable: correct the
plan format or the request without bypassing planning restrictions.
'''

PLAN_SCHEMA = {"type": "function", "function": {
    "name": "submit_plan", "description": "Save the structured plan and wait for explicit operator approval.",
    "parameters": {"type": "object", "additionalProperties": False,
        "required": ["goal", "steps", "risks", "assumptions", "done_criteria"],
        "properties": {
            "goal": {"type": "string"},
            "steps": {"type": "array", "minItems": 1, "maxItems": 30, "items": {
                "type": "object", "additionalProperties": False, "required": ["description", "tools"],
                "properties": {"description": {"type": "string"}, "tools": {
                    "type": "array", "minItems": 1, "items": {"type": "string", "enum": sorted(TOOLS)}}}}},
            **{key: {"type": "array", "items": {"type": "string"}} for key in ("risks", "assumptions", "done_criteria")}
        }}}}


def validate_plan(value: object) -> dict:
    keys = {"goal", "steps", "risks", "assumptions", "done_criteria"}
    if not isinstance(value, dict) or set(value) != keys:
        raise PlanError("Plan requires exactly goal, steps, risks, assumptions and done_criteria")
    def text(item):
        if not isinstance(item, str) or not item.strip() or len(item) > 4000:
            raise PlanError("Plan text must be non-empty and at most 4000 characters")
    text(value["goal"])
    if not isinstance(value["steps"], list) or not 1 <= len(value["steps"]) <= 30:
        raise PlanError("Plan needs 1 to 30 ordered steps")
    for step in value["steps"]:
        if not isinstance(step, dict) or set(step) != {"description", "tools"}:
            raise PlanError("Every step requires description and tools")
        text(step["description"])
        tools = step["tools"]
        if (not isinstance(tools, list) or not 1 <= len(tools) <= len(TOOLS) or
                any(not isinstance(t, str) or t not in TOOLS for t in tools) or len(set(tools)) != len(tools)):
            raise PlanError("Step tools must be distinct supported tool names")
    for key in ("risks", "assumptions", "done_criteria"):
        items = value[key]
        if not isinstance(items, list) or len(items) > 30 or (key == "done_criteria" and not items):
            raise PlanError("Plan lists need at most 30 entries; done_criteria must not be empty")
        for item in items:
            text(item)
    if len(json.dumps(value, ensure_ascii=False)) > 40000:
        raise PlanError("Plan exceeds the 40000-character limit")
    return copy.deepcopy(value)
