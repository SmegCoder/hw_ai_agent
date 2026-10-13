from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path

from .tools import ToolError, protected, is_test
from .policy import Action, ApprovalRequest
from .filesystem import (open_directory, open_read_file, signature, same_signature,
                         capture_target, verify_target, write_bytes_target, remove_target)


MAX_FILES = 2000
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TREE_BYTES = 20 * 1024 * 1024
REPORTS = {'demo': 'demo-report.json', 'plan-demo': 'demo-report.json',
           'workflow': 'workflow-report.json', 'tasks-demo': 'tasks-report.json',
           'rollback-demo': 'rollback-report.json'}


class JournalError(RuntimeError):
    pass


class JournalConflict(JournalError):
    pass


class JournalUncertain(JournalError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def changed_paths(before, after):
    return sorted(p for p in set(before['files']) | set(after['files']) if before['files'].get(p) != after['files'].get(p))


def valid_path(path):
    if (not isinstance(path, str) or not path or '\x00' in path or Path(path).is_absolute()
            or '..' in Path(path).parts or Path(path).as_posix() != path or path == '.' or protected(Path(path))):
        raise JournalError('Invalid saved journal path')


def valid_entry(entry):
    if entry is None:
        return
    if not isinstance(entry, dict):
        raise JournalError('Invalid saved file state')
    if entry.get('kind') == 'file':
        if (set(entry) != {'kind','sha256','mode','size'} or not isinstance(entry['sha256'],str)
                or not re.fullmatch('[a-f0-9]{64}',entry['sha256']) or type(entry['mode']) is not int
                or not 0 <= entry['mode'] <= 0o777 or type(entry['size']) is not int or not 0 <= entry['size'] <= MAX_FILE_BYTES):
            raise JournalError('Invalid saved file hash/size/mode')
    elif entry.get('kind') == 'symlink':
        if set(entry) != {'kind','target'} or not isinstance(entry['target'],str):
            raise JournalError('Invalid saved symlink state')
    else:
        raise JournalError('Unsupported saved file type')


class ChangeJournal:
    def __init__(self, store, session_id, runtime):
        self.store, self.id, self.runtime = store, session_id, runtime
        self.db = store.db
        self.excluded_report = REPORTS.get(store.get(session_id).metadata.get('kind'))
        self.active_call = None
        self.reviewed_tree = None

    def _scan(self):
        files, directories, blobs = {}, [], {}
        total = 0
        root = self.runtime.root
        with open_directory(root, root) as fd:
            info = os.fstat(fd)
            identity = [info.st_dev, info.st_ino]
            for current, folders, names, current_fd in os.fwalk('.', follow_symlinks=False, dir_fd=fd):
                relative = Path(current)
                if relative.as_posix() != '.':
                    directories.append(relative.as_posix())
                kept = []
                for name in sorted(folders):
                    path = (relative / name).as_posix()
                    if protected(Path(path)) or any(Path(path).is_relative_to(Path(d)) for d in self.runtime.policy.directory_deny):
                        continue
                    allowed = self.runtime.policy.directory_allowed(path)
                    ancestor = any(Path(a).is_relative_to(Path(path)) for a in self.runtime.policy.directory_allow)
                    if not allowed and not ancestor:
                        continue
                    entry = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                    if stat.S_ISLNK(entry.st_mode):
                        target = os.readlink(name, dir_fd=current_fd)
                        files[path] = {'kind': 'symlink', 'target': target}
                    elif stat.S_ISDIR(entry.st_mode):
                        kept.append(name)
                    else:
                        raise JournalError('Unsupported workspace directory entry')
                folders[:] = kept
                for name in sorted(names):
                    path = (relative / name).as_posix()
                    if protected(Path(path)) or path == self.excluded_report or not self.runtime.policy.directory_allowed(path):
                        continue
                    entry = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                    if stat.S_ISLNK(entry.st_mode):
                        files[path] = {'kind': 'symlink', 'target': os.readlink(name, dir_fd=current_fd)}
                        continue
                    if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
                        raise JournalError('Journal requires regular files without hard links')
                    if self.runtime.policy.decide('read_file', path=path).action != Action.ALLOW:
                        raise JournalError('Journal snapshot needs a permitted read: ' + path)
                    if entry.st_size > MAX_FILE_BYTES:
                        raise JournalError('Journal file exceeds snapshot limit')
                    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current_fd)
                    try:
                        before = signature(os.fstat(file_fd))
                        if not same_signature(signature(entry), before):
                            raise JournalConflict('File replaced during snapshot: ' + path)
                        with os.fdopen(file_fd, 'rb', closefd=False) as stream:
                            data = stream.read(MAX_FILE_BYTES + 1)
                        if len(data) > MAX_FILE_BYTES or not same_signature(before, signature(os.fstat(file_fd))):
                            raise JournalConflict('File changed during snapshot: ' + path)
                    finally:
                        os.close(file_fd)
                    total += len(data)
                    if total > MAX_TREE_BYTES or len(files) >= MAX_FILES:
                        raise JournalError('Workspace exceeds journal snapshot limits')
                    sha = digest(data)
                    blobs[sha] = data
                    files[path] = {'kind': 'file', 'sha256': sha, 'mode': stat.S_IMODE(entry.st_mode), 'size': len(data)}
                if len(files) + len(directories) > MAX_FILES:
                    raise JournalError('Workspace exceeds snapshot path limit')
            named = root.lstat()
            if [named.st_dev,named.st_ino] != identity or not stat.S_ISDIR(named.st_mode):
                raise JournalConflict('Workspace root replaced during snapshot')
        return {'root': identity, 'dirs': sorted(directories), 'files': files}, blobs

    def capture(self):
        try:
            first, _ = self._scan()
            second, blobs = self._scan()
        except (OSError, UnicodeError) as exc:
            raise JournalConflict('Cannot capture a stable workspace tree') from None
        if first != second:
            raise JournalConflict('Workspace changed during snapshot')
        return second, blobs

    def _save_image(self, tree, blobs):
        for sha, data in blobs.items():
            self.db.execute('INSERT OR IGNORE INTO journal_blobs VALUES (?,?)', (sha, data))
        rev = self.db.execute('SELECT COALESCE(MAX(revision),0)+1 FROM journal_images WHERE session_id=?', (self.id,)).fetchone()[0]
        self.db.execute('INSERT INTO journal_images VALUES (?,?,?)', (self.id, rev, encoded(tree)))
        return rev

    def image(self, revision):
        row = self.db.execute('SELECT tree FROM journal_images WHERE session_id=? AND revision=?', (self.id, revision)).fetchone()
        try:
            if not row:
                raise ValueError()
            tree = json.loads(row[0])
            if set(tree) != {'root', 'dirs', 'files'} or not isinstance(tree['files'], dict):
                raise ValueError()
            if not isinstance(tree['root'], list) or len(tree['root']) != 2 or any(type(i) is not int for i in tree['root']):
                raise ValueError()
            for path in tree['files']:
                valid_path(path)
                valid_entry(tree['files'][path])
            for path in tree['dirs']:
                valid_path(path)
            return tree
        except (ValueError, TypeError, KeyError):
            raise JournalError('Invalid saved workspace snapshot') from None

    def blob(self, entry):
        if entry is None:
            return b''
        if entry.get('kind') != 'file':
            raise JournalError('Rollback of non-regular files is not supported')
        sha = entry.get('sha256')
        row = self.db.execute('SELECT data FROM journal_blobs WHERE sha256=?', (sha,)).fetchone()
        if not row or type(entry.get('size')) is not int or not 0 <= entry['size'] <= MAX_FILE_BYTES:
            raise JournalError('Invalid journal content')
        data = bytes(row[0])
        if len(data) != entry['size'] or digest(data) != sha or type(entry.get('mode')) is not int or not 0 <= entry['mode'] <= 0o777:
            raise JournalError('Journal content hash or permissions are invalid')
        return data

    def head(self):
        row = self.db.execute('SELECT * FROM journals WHERE session_id=?', (self.id,)).fetchone()
        if not row or row['status'] not in {'active', 'rollback_pending', 'rolled_back'}:
            raise JournalError('Session has no valid change journal')
        self.image(row['revision'])
        return row

    def start(self):
        if self.db.execute('SELECT 1 FROM journals WHERE session_id=?', (self.id,)).fetchone():
            return self.head()
        state = self.store.get(self.id)
        if state.rounds or self.store.calls(self.id):
            raise JournalError('Cannot reconstruct the original state of an existing run')
        tree, blobs = self.capture()
        with self.store.transaction():
            rev = self._save_image(tree, blobs)
            self.db.execute('INSERT INTO journals VALUES (?,?,\'active\',NULL)', (self.id, rev))
            self.store._event(self.id, {'event': 'journal_started', 'revision': rev})
        return self.head()

    def entries(self):
        rows = self.db.execute('SELECT * FROM journal_entries WHERE session_id=? ORDER BY path', (self.id,)).fetchall()
        result = {}
        for row in rows:
            valid_path(row['path'])
            try:
                result[row['path']] = {'before': json.loads(row['before_state']), 'after': json.loads(row['after_state'])}
                valid_entry(result[row['path']]['before']);valid_entry(result[row['path']]['after'])
            except (ValueError, TypeError):
                raise JournalError('Invalid saved file change') from None
        return result

    def _replace_entries(self, entries):
        self.db.execute('DELETE FROM journal_entries WHERE session_id=?', (self.id,))
        for path, entry in entries.items():
            valid_path(path);valid_entry(entry['before']);valid_entry(entry['after'])
            self.db.execute('INSERT INTO journal_entries VALUES (?,?,?,?)', (self.id, path, encoded(entry['before']), encoded(entry['after'])))

    def _own_changes(self, before, after, entries):
        result = copy.deepcopy(entries)
        for path in changed_paths(before, after):
            original = result[path]['before'] if path in result else before['files'].get(path)
            result[path] = {'before': original, 'after': after['files'].get(path)}
        self._replace_entries(result)

    def latest_action(self, key):
        return self.db.execute('SELECT * FROM journal_actions WHERE session_id=? AND call_key=? ORDER BY ordinal DESC LIMIT 1', (self.id, key)).fetchone()

    @contextmanager
    def action(self, name, arguments, path):
        if not self.active_call:
            raise JournalError('Journal mutation requires a registered tool call')
        head = self.head()
        if head['status'] != 'active':
            raise JournalError('Rollback must finish before new actions')
        expected = self.image(head['revision'])
        current, blobs = self.capture()
        if current != expected:
            raise JournalConflict('JOURNAL_CONFLICT: workspace differs from last checkpoint; inspect changes')
        if self.excluded_report is not None and path == self.excluded_report:
            raise JournalConflict('Generated demo reports cannot be changed by model tools')
        key = self.active_call['key']
        prior = self.entries()
        with self.store.transaction():
            rev = self._save_image(current, blobs)
            ordinal = self.db.execute('SELECT COALESCE(MAX(ordinal),0)+1 FROM journal_actions WHERE session_id=? AND call_key=?', (self.id, key)).fetchone()[0]
            self.db.execute('INSERT INTO journal_actions VALUES (?,?,?,?,NULL,\'started\',?,NULL)', (self.id, key, ordinal, rev, encoded(prior)))
            self.store._event(self.id, {'event': 'journal_action_started', 'call': key, 'revision': rev})
        try:
            yield
        finally:
            try:
                after, after_blobs = self.capture()
            except JournalError:
                raise JournalUncertain('Cannot checkpoint the post-action tree; explicit recovery required') from None
            changed = changed_paths(current, after)
            allowed = not changed if name == 'Bash' else set(changed) <= {path}
            if current['root'] != after['root']:
                allowed = False
            if name == 'write_file' and changed:
                entry = after['files'].get(path)
                allowed = allowed and entry is not None and entry.get('kind') == 'file' and entry.get('sha256') == digest(arguments['content'].encode('utf-8'))
                if current['files'].get(path):
                    allowed = allowed and entry.get('mode') == current['files'][path].get('mode')
            if name == 'delete_file' and changed:
                allowed = allowed and path not in after['files']
            # New parent directories from a write are allowed; other changes need attribution.
            created_dirs = set(after['dirs']) - set(current['dirs'])
            removed_dirs = set(current['dirs']) - set(after['dirs'])
            parents = {p.as_posix() for p in Path(path or '.').parents if p.as_posix() != '.'}
            allowed = allowed and not removed_dirs and created_dirs <= parents
            if not allowed:
                raise JournalUncertain('JOURNAL_UNCERTAIN: unexpected tree changes; explicit attribution required')
            with self.store.transaction():
                post = self._save_image(after, after_blobs)
                self._own_changes(current, after, prior)
                self.db.execute('UPDATE journals SET revision=? WHERE session_id=?', (post, self.id))
                self.db.execute('UPDATE journal_actions SET post_revision=?,state=\'completed\' WHERE session_id=? AND call_key=? AND ordinal=?', (post, self.id, key, ordinal))
                self.store._event(self.id, {'event': 'journal_action_completed', 'call': key, 'revision': post, 'paths': changed,
                    'changes': {p:{'before':current['files'].get(p),'after':after['files'].get(p)} for p in changed},
                    'directories_added':sorted(created_dirs)})

    def diff(self, entries=None):
        parts = []
        for path, entry in (entries if entries is not None else self.entries()).items():
            if entry['before'] == entry['after']:
                continue
            if any(state is not None and state.get('kind') != 'file' for state in (entry['before'],entry['after'])):
                parts.append(f"File {path}: {encoded(entry['before'])} -> {encoded(entry['after'])}\n")
                continue
            try:
                old = self.blob(entry['before']).decode('utf-8')
                new = self.blob(entry['after']).decode('utf-8')
            except UnicodeError:
                parts.append(f'File {path}: {encoded(entry["before"])} -> {encoded(entry["after"])}\n')
                continue
            lines = difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                       fromfile='a/'+path if entry['before'] is not None else '/dev/null',
                                       tofile='b/'+path if entry['after'] is not None else '/dev/null')
            if not old and not new and (entry['before'] is None or entry['after'] is None):
                parts.append(f'Empty file {path}: {"created" if entry["before"] is None else "deleted"}\n')
            for line in lines:
                parts.append(line if line.endswith('\n') else line+'\n\\ No newline at end of file\n')
            if entry['before'] and entry['after'] and entry['before'].get('mode') != entry['after'].get('mode'):
                parts.append(f'Permissions changed: {path}\n')
        return ''.join(parts) or '(no file changes)'

    def recovery_view(self, item):
        action = self.latest_action(item['key'])
        before = self.image(action['pre_revision'] if action else self.head()['revision'])
        current, blobs = self.capture()
        if current['root'] != before['root']:
            raise JournalConflict('Workspace root identity changed; attribution refused')
        delta = {p: {'before': before['files'].get(p), 'after': current['files'].get(p)} for p in changed_paths(before, current)}
        # Keep blobs in memory for a read-only preview; commit only on a decision.
        text = self._preview_delta(delta, blobs)
        if current['dirs'] != before['dirs']:
            text += '\nDirectories changed: ' + encoded({'before':before['dirs'],'after':current['dirs']})
        return {'diff': text, 'paths': list(delta), 'changed': current != before,
                '_before': before, '_current': current, '_blobs': blobs, '_action': action}

    def _preview_delta(self, entries, blobs):
        parts = []
        for path, entry in entries.items():
            def data(state):
                if state is None:return ''
                if state.get('kind') != 'file':return encoded(state)+'\n'
                raw = blobs.get(state['sha256'])
                return (raw if raw is not None else self.blob(state)).decode('utf-8', errors='replace')
            parts.extend(difflib.unified_diff(data(entry['before']).splitlines(True), data(entry['after']).splitlines(True), fromfile='before/'+path, tofile='observed/'+path))
        return ''.join(parts) or '(no file changes)'

    def resolve_changes(self, item, ownership):
        view = self.recovery_view(item)
        if self.reviewed_tree is not None and view['_current'] != self.reviewed_tree:
            raise JournalConflict('New changes appeared after preview; review attribution again')
        if view['changed'] and ownership not in {'own', 'external'}:
            raise JournalError('Explicit change attribution required: --changes own|external')
        action = view['_action']
        prior = json.loads(action['prior_entries']) if action else self.entries()
        with self.store.transaction():
            post = self._save_image(view['_current'], view['_blobs'])
            if ownership == 'own':
                self._own_changes(view['_before'], view['_current'], prior)
                self.db.execute('UPDATE journals SET revision=? WHERE session_id=?', (post, self.id))
            elif ownership == 'external':
                self._replace_entries(prior)
                if action:self.db.execute('UPDATE journals SET revision=? WHERE session_id=?', (action['pre_revision'], self.id))
            if action:
                self.db.execute('UPDATE journal_actions SET state=?,ownership=?,post_revision=? WHERE session_id=? AND call_key=? AND ordinal=?',
                                ('resolved', ownership or 'none', post, self.id, item['key'], action['ordinal']))
            self.store._event(self.id, {'event': 'journal_attribution', 'call': item['key'], 'ownership': ownership or 'none', 'paths': view['paths'],
                'changes': {p:{'before':view['_before']['files'].get(p),'after':view['_current']['files'].get(p)} for p in view['paths']}})
        self.reviewed_tree = None

    def rollback(self, after_file=None):
        with self.store.lock(self.id):
            return self._rollback(after_file)

    def _rollback(self, after_file=None):
        head = self.head()
        if any(i['state'] in {'running', 'uncertain'} for i in self.store.calls(self.id, True)):
            raise JournalError('Resolve the uncertain tool call and its changes before rollback')
        if head['status'] == 'rolled_back':
            return {'status': 'already_rolled_back', 'paths': []}
        entries = self.entries()
        current, _ = self.capture()
        if current['root'] != self.image(head['revision'])['root']:
            raise JournalConflict('Workspace root changed; rollback refused')
        pending = head['status'] == 'rollback_pending'
        started = {e.get('path') for e in self.store.get(self.id).events if e.get('event')=='rollback_file_started'}
        # Every path, every content hash and every policy is checked before any mutation.
        operations = []
        for path, entry in entries.items():
            observed = current['files'].get(path)
            if observed != entry['after'] and not (pending and path in started and observed == entry['before']):
                raise JournalConflict('ROLLBACK_CONFLICT: externally modified file: '+path)
            original = entry['before']
            if original:self.blob(original)
            resolved = self.runtime._safe_path(path)
            if resolved.relative_to(self.runtime.root).as_posix() != path or is_test(Path(path)):
                raise JournalError('Rollback path is protected or noncanonical')
            tool = 'delete_file' if original is None else 'write_file'
            decision = self.runtime.policy.decide(tool, path=path)
            if decision.action == Action.DENY:
                raise JournalError('POLICY_DENIED: rollback '+path)
            if entry['before'] == entry['after'] or pending and observed == original:
                continue
            target = capture_target(self.runtime.root, resolved, path)
            operations.append((path, entry, tool, target))
        for path, entry, tool, target in operations:
            if tool == 'delete_file' or self.runtime.mode == 'safe':
                fingerprint = digest(encoded({'path':path,'entry':entry,'signature':target.file_signature}).encode())
                preview = self.diff({path: {'before':entry['after'], 'after':entry['before']}})[:4000]
                request = ApprovalRequest(tool, path, 'Explicit rollback of this session', preview, fingerprint)
                if self.runtime.approve is None or self.runtime.approve(request) is not True:
                    raise JournalError('Rollback declined; no files were changed')
        # An approval can take minutes. Recheck the entire prepared set afterwards.
        refreshed, _ = self.capture()
        if refreshed != current:
            raise JournalConflict('ROLLBACK_CONFLICT: workspace changed while confirmations were pending')
        for _, _, _, target in operations:verify_target(target)
        for path, _, tool, _ in operations:
            if self.runtime.policy.decide(tool,path=path).action==Action.DENY:
                raise JournalError('POLICY_DENIED: rollback policy changed during confirmation')
        with self.store.transaction():
            self.db.execute('UPDATE journals SET status=\'rollback_pending\',rollback_plan=? WHERE session_id=?', (encoded(entries), self.id))
            self.store._event(self.id, {'event':'rollback_started','paths':list(entries)})
        for path, entry, tool, target in operations:
            # Persist intent before touching a file; interrupted rollback can resume by hashes.
            with self.store.transaction():self.store._event(self.id, {'event':'rollback_file_started','path':path})
            if tool == 'delete_file':remove_target(target, self.runtime.trash_dir)
            else:write_bytes_target(target, self.blob(entry['before']), mode=entry['before']['mode'])
            if after_file:after_file(path)
            with self.store.transaction():self.store._event(self.id, {'event':'rollback_file_completed','path':path})
        final, blobs = self.capture()
        if any(final['files'].get(p) != e['before'] for p,e in entries.items()):
            raise JournalConflict('Rollback result differs from original state; journal remains pending')
        with self.store.transaction():
            rev = self._save_image(final, blobs)
            self.db.execute('UPDATE journals SET status=\'rolled_back\',revision=? WHERE session_id=?', (rev,self.id))
            self.db.execute('UPDATE sessions SET status=\'rolled_back\',error=NULL WHERE id=?', (self.id,))
            self.store._event(self.id, {'event':'rollback_completed','paths':list(entries)})
        return {'status':'rolled_back','paths':list(entries)}
