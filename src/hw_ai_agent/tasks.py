from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
from pathlib import PurePosixPath

from .planning import PLAN_SCHEMA, validate_plan, PlanError
from .tools import ToolError
from .policy import Action
from .filesystem import open_read_file, open_directory
from .budget import TRUNCATED


class TaskError(ToolError):
    pass


TASK_INSTRUCTIONS = '''
Task mode is enabled. This overrides the instruction to finish immediately after
passing tests: first complete the current task with complete_task(task_id).
Every submit_plan step must include id, description, tools, depends_on and
verification. IDs must be unique. Dependencies must form an acyclic graph.
Verification is one of {"kind":"file_written","path":"relative/file.py"},
{"kind":"file_read","path":"relative/file.py"}, or
{"kind":"file_absent","path":"relative/unused.py"}, or
{"kind":"tests_passed","command":"python -m unittest discover -v","cwd":"."}.
Use file_read for investigation, file_written for a code change, file_absent
for deletion, tests_passed
for testing. Include the corresponding tool in each task's tools.
Only execute the CURRENT_TASK selected by runtime; you cannot select a task.
complete_task must be the LAST call in its batch. It checks actual tool evidence;
your own text is not evidence. Correct errors and retry verification if needed.
Use fail_task(task_id, reason) only if you cannot finish the current task.
After ALL_TASKS_COMPLETED return your final answer without further tools.
Do not reuse tool_call_id across tasks or between planning and execution.
'''

TASK_PLAN_SCHEMA = copy.deepcopy(PLAN_SCHEMA)
step = TASK_PLAN_SCHEMA['function']['parameters']['properties']['steps']['items']
step['required'] += ['id', 'depends_on', 'verification']
step['properties'].update({
    'id': {'type': 'string', 'pattern': '^[a-z][a-z0-9_]{0,63}$'},
    'depends_on': {'type': 'array', 'maxItems': 29, 'items': {'type': 'string'}},
    'verification': {'anyOf': [
        {'type': 'object', 'additionalProperties': False, 'required': ['kind', 'path'],
         'properties': {'kind': {'type': 'string', 'enum': ['file_written', 'file_read', 'file_absent']}, 'path': {'type': 'string'}}},
        {'type': 'object', 'additionalProperties': False, 'required': ['kind', 'command', 'cwd'],
         'properties': {'kind': {'type': 'string', 'enum': ['tests_passed']}, 'command': {'type': 'string'}, 'cwd': {'type': 'string'}}}
    ]}
})
TASK_SCHEMAS = [{'type': 'function', 'function': {
    'name': name, 'description': description,
    'parameters': {'type': 'object', 'additionalProperties': False,
                   'required': list(props), 'properties': {p: {'type': 'string'} for p in props}}
}} for name, description, props in [
    ('complete_task', 'Verify the current task against actual tool results; must be the last call in the batch.', ['task_id']),
    ('fail_task', 'Record failure of the current task and block its dependants; must be the last call in the batch.', ['task_id', 'reason'])
]]


def relative_path(value):
    if not isinstance(value, str) or not value or len(value) > 4000 or '\x00' in value:
        raise PlanError('Verification path must be a non-empty relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise PlanError('Verification paths must be canonical relative paths without ..')
    return value


def test_command(value):
    if not isinstance(value, str) or len(value) > 4000 or any(c in value for c in '\n\r\x00$`;|&<>'):
        raise PlanError('Verification requires a simple unittest command')
    try:
        argv = shlex.split(value)
    except ValueError:
        raise PlanError('Invalid verification command') from None
    if len(argv) < 3 or argv[0] not in {'python', 'python3'} or argv[1:3] != ['-m', 'unittest']:
        raise PlanError('Verification requires python[3] -m unittest')
    return shlex.join(argv)


def validate_task_plan(value):
    if not isinstance(value, dict) or not isinstance(value.get('steps'), list):
        raise PlanError('Task plan needs steps')
    base = copy.deepcopy(value)
    for step in base['steps']:
        if not isinstance(step, dict) or set(step) != {'id', 'description', 'tools', 'depends_on', 'verification'}:
            raise PlanError('Each task requires id, description, tools, depends_on and verification')
    base['steps'] = [{'description': s['description'], 'tools': s['tools']} for s in base['steps']]
    validate_plan(base)
    ids = []
    for step in value['steps']:
        identifier = step['id']
        if not isinstance(identifier, str) or not re.fullmatch('[a-z][a-z0-9_]{0,63}', identifier) or identifier in ids:
            raise PlanError('Task IDs must be distinct lowercase identifiers, at most 64 characters')
        ids.append(identifier)
        dependencies = step['depends_on']
        if (not isinstance(dependencies, list) or len(dependencies) > 29 or
                any(not isinstance(d, str) for d in dependencies) or len(set(dependencies)) != len(dependencies)):
            raise PlanError('Task dependencies must be distinct IDs')
        verification = step['verification']
        if not isinstance(verification, dict):
            raise PlanError('Task verification must be an object')
        kind = verification.get('kind')
        if kind in {'file_written', 'file_read', 'file_absent'}:
            if set(verification) != {'kind', 'path'}:
                raise PlanError('File verification requires exactly kind and path')
            relative_path(verification['path'])
            required = {'file_written': 'write_file', 'file_read': 'read_file', 'file_absent': 'delete_file'}[kind]
        elif kind == 'tests_passed':
            if set(verification) != {'kind', 'command', 'cwd'}:
                raise PlanError('Test verification requires exactly kind, command and cwd')
            test_command(verification['command'])
            relative_path(verification['cwd'])
            required = 'Bash'
        else:
            raise PlanError('Unknown task verification kind')
        if required not in step['tools']:
            raise PlanError('Task tools must include the tool needed for verification')
    graph = {s['id']: s['depends_on'] for s in value['steps']}
    visited, active = set(), set()
    def visit(identifier):
        if identifier not in graph or identifier in active:
            raise PlanError('Task dependencies contain an unknown ID or a cycle')
        if identifier in visited:
            return
        active.add(identifier)
        for dependency in graph[identifier]:
            visit(dependency)
        active.remove(identifier)
        visited.add(identifier)
    for identifier in ids:
        visit(identifier)
    if len(json.dumps(value, ensure_ascii=False)) > 40000:
        raise PlanError('Task plan exceeds 40000 characters')
    return copy.deepcopy(value)


def passing_tests(content):
    return (content.startswith('exit_code=0\n') and re.search(r'\bRan [1-9][0-9]* tests? in ', content) is not None
            and re.search(r'^OK(?: \([^\n]*\))?$', content, re.MULTILINE) is not None
            and not content.endswith(TRUNCATED) and '\noutput truncated: command stopped at output limit' not in content)


def candidate_evidence(task, calls):
    verification = task['verification']
    kind = verification['kind']
    for item in reversed(calls):
        if item.get('task_id') != task['id']:
            continue
        function = item['call']['function']
        name = 'Bash' if function['name'] == 'bash' else function['name']
        try:
            arguments = json.loads(function['arguments'])
            if not isinstance(arguments, dict):
                continue
            if kind == 'tests_passed':
                match = (name == 'Bash' and test_command(arguments.get('command')) == test_command(verification['command'])
                         and arguments.get('cwd', '.') == verification['cwd'])
            else:
                match = (name == {'file_written': 'write_file', 'file_read': 'read_file', 'file_absent': 'delete_file'}[kind]
                         and arguments.get('path') == verification['path'])
        except (ValueError, TypeError):
            continue
        if not match:
            continue
        # A newer failed check must never be hidden by an older successful one.
        if item['state'] != 'completed' or not item['result']:
            raise TaskError('TASK_NOT_VERIFIED: latest matching tool call did not succeed')
        output = item['result']['content']
        if kind == 'tests_passed':
            if not passing_tests(output):
                raise TaskError('TASK_NOT_VERIFIED: latest test run did not pass completely')
            expected = output
        else:
            expected = arguments['content'] if kind == 'file_written' else 'absent' if kind == 'file_absent' else output
        return item, hashlib.sha256(expected.encode('utf-8')).hexdigest()
    raise TaskError('TASK_NOT_VERIFIED: no matching successful tool result for this task')


def verify_task(task, calls, runtime):
    item, digest = candidate_evidence(task, calls)
    verification = task['verification']
    if verification['kind'] != 'tests_passed':
        path = runtime._safe_path(verification['path'])
        relative = path.relative_to(runtime.root).as_posix()
        if relative != verification['path'] or runtime.policy.decide('read_file', path=relative).action != Action.ALLOW:
            raise TaskError('TASK_NOT_VERIFIED: verification needs a permitted canonical file read')
        if verification['kind'] == 'file_absent':
            with open_directory(runtime.root, path.parent) as fd:
                try:
                    os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    actual = b'absent'
                else:
                    raise TaskError('TASK_NOT_VERIFIED: file still exists')
        elif verification['kind'] == 'file_read':
            actual = runtime.read_file(relative).encode('utf-8')
        else:
            with open_read_file(runtime.root, path) as fd:
                if os.fstat(fd).st_size > runtime.max_file_chars * 4:
                    raise TaskError('TASK_NOT_VERIFIED: file exceeds verification size limit')
                with os.fdopen(fd, 'rb', closefd=False) as stream:
                    actual = stream.read(runtime.max_file_chars * 4 + 1)
        if hashlib.sha256(actual).hexdigest() != digest:
            raise TaskError('TASK_NOT_VERIFIED: current file differs from recorded tool result')
    return {'kind': verification['kind'], 'call_key': item['key'], 'sha256': digest}


def validate_proof(task, calls):
    proof = task.get('proof')
    if (not isinstance(proof, dict) or set(proof) != {'kind', 'call_key', 'sha256'}
            or proof['kind'] != task['verification']['kind']):
        raise TaskError('Invalid saved task proof')
    item, digest = candidate_evidence(task, calls)
    if item['key'] != proof['call_key'] or digest != proof['sha256']:
        raise TaskError('Saved task proof differs from actual tool evidence')


def checklist(tasks):
    marks = {'pending': '[ ]', 'running': '[>]', 'completed': '[x]', 'failed': '[!]', 'blocked': '[-]'}
    return '\n'.join(f"{marks[t['state']]} {t['id']} — {t['description']}" +
                     (f" (зависит от: {', '.join(t['depends_on'])})" if t['depends_on'] else '') for t in tasks)
