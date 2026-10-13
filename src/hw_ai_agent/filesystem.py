from __future__ import annotations

import os
import hashlib
import json
import stat
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path


class TargetChangedError(OSError):
    pass


def signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink, info.st_mode)


def same_signature(observed: tuple, expected: tuple) -> bool:
    # macOS provenance attributes can update ctime without changing file data.
    # Identity, size, mtime, links and mode must match; content is checked separately.
    return observed[:4] == expected[:4] and observed[5:] == expected[5:]


def file_digest(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 65536):
        digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def open_directory(root: Path, path: Path, directories=()):
    expected = {p: (device, inode) for p, device, inode in directories}
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = None
    try:
        parent_fd = os.open(root, flags)
        current = root
        for component in (None, *path.relative_to(root).parts):
            if component is not None:
                next_fd = os.open(component, flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = next_fd
                current = current / component
            info = os.fstat(parent_fd)
            if current in expected and (info.st_dev, info.st_ino) != expected[current]:
                raise TargetChangedError("A checked parent directory changed")
        yield parent_fd
    except OSError as exc:
        if isinstance(exc, TargetChangedError):
            raise
        raise TargetChangedError("Cannot open a stable directory") from None
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


@contextmanager
def open_read_file(root: Path, path: Path, directories=()):
    file_fd = None
    try:
        with open_directory(root, path.parent, directories) as parent_fd:
            file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise TargetChangedError("Read target must be a regular file without hard links")
            yield file_fd
    except OSError as exc:
        if isinstance(exc, TargetChangedError):
            raise
        raise TargetChangedError("Cannot open a stable read target") from None
    finally:
        if file_fd is not None:
            os.close(file_fd)


@dataclass(frozen=True)
class WriteTarget:
    root: Path
    path: Path
    original: str
    file_signature: tuple[int, ...] | None
    directories: tuple[tuple[Path, int, int], ...]
    file_digest: str | None


def capture_target(root: Path, path: Path, original: str) -> WriteTarget:
    directories = []
    current = root
    for component in (None, *path.relative_to(root).parts[:-1]):
        if component is not None:
            current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if not stat.S_ISDIR(info.st_mode):
            raise TargetChangedError("A parent is not a regular directory")
        directories.append((current, info.st_dev, info.st_ino))
    try:
        info = path.lstat()
    except FileNotFoundError:
        file_signature = None
        digest = None
    else:
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TargetChangedError("Target must be a regular file without hard links")
        file_signature = signature(info)
        with open_read_file(root, path, directories) as descriptor:
            if not same_signature(signature(os.fstat(descriptor)), file_signature):
                raise TargetChangedError("File changed before capture")
            digest = file_digest(descriptor)
            if not same_signature(signature(os.fstat(descriptor)), file_signature):
                raise TargetChangedError("File changed during capture")
    return WriteTarget(root, path, original, file_signature, tuple(directories), digest)


def verify_target(target: WriteTarget) -> None:
    try:
        if (target.root / target.original).resolve() != target.path:
            raise TargetChangedError("The requested path now resolves to another target")
        for directory, device, inode in target.directories:
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (device, inode):
                raise TargetChangedError("A parent directory changed")
        try:
            info = target.path.lstat()
        except FileNotFoundError:
            if target.file_signature is not None:
                raise TargetChangedError("The approved file disappeared")
        else:
            if target.file_signature is None or not same_signature(signature(info), target.file_signature):
                raise TargetChangedError("The approved file changed")
            with open_read_file(target.root, target.path, target.directories) as descriptor:
                if (not same_signature(signature(os.fstat(descriptor)), target.file_signature) or
                        file_digest(descriptor) != target.file_digest or
                        not same_signature(signature(os.fstat(descriptor)), target.file_signature)):
                    raise TargetChangedError("The approved contents changed")
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, TargetChangedError):
            raise
        raise TargetChangedError("Cannot verify the approved target") from None


def write_target(target: WriteTarget, content: str) -> None:
    write_bytes_target(target, content.encode("utf-8"))


def write_bytes_target(target: WriteTarget, data: bytes, *, mode: int | None = None) -> None:
    verify_target(target)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(target.root, directory_flags)
    file_fd = None
    try:
        expected_dirs = {p: (device, inode) for p, device, inode in target.directories}
        root_info = os.fstat(directory_fd)
        if (root_info.st_dev, root_info.st_ino) != expected_dirs[target.root]:
            raise TargetChangedError("Workspace root changed")
        current = target.root
        for component in target.path.relative_to(target.root).parts[:-1]:
            current = current / component
            if current not in expected_dirs:
                try:
                    os.mkdir(component, dir_fd=directory_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            info = os.fstat(next_fd)
            if current in expected_dirs and (info.st_dev, info.st_ino) != expected_dirs[current]:
                os.close(next_fd)
                raise TargetChangedError("Parent directory replaced during execution")
            os.close(directory_fd)
            directory_fd = next_fd
        # No pathname-based truncate: inspect the opened inode before changing it.
        flags = (os.O_WRONLY if target.file_signature is None else os.O_RDWR) | os.O_NOFOLLOW | os.O_NONBLOCK
        if target.file_signature is None:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            file_fd = os.open(target.path.name, flags, 0o666, dir_fd=directory_fd)
        except OSError:
            raise TargetChangedError("Cannot open the same approved target safely") from None
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TargetChangedError("Opened target is not a regular, unlinked file")
        if target.file_signature is not None:
            if (not same_signature(signature(info), target.file_signature) or
                    file_digest(file_fd) != target.file_digest or
                    not same_signature(signature(os.fstat(file_fd)), target.file_signature)):
                raise TargetChangedError("File replaced or edited during execution")
        if target.file_signature is not None:
            os.ftruncate(file_fd, 0)
            os.lseek(file_fd, 0, os.SEEK_SET)
        view = memoryview(data)
        while view:
            count = os.write(file_fd, view)
            view = view[count:]
        if mode is not None:
            os.fchmod(file_fd, mode)
        os.fsync(file_fd)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def remove_target(target: WriteTarget, trash_dir: Path) -> str | None:
    verify_target(target)
    if target.file_signature is None:
        return None
    if trash_dir.resolve().is_relative_to(target.root):
        raise TargetChangedError("Recovery directory must be outside workspace")
    trash_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(target.root, flags)
    trash_fd = backup_fd = None
    try:
        expected = {p: (device, inode) for p, device, inode in target.directories}
        info = os.fstat(parent_fd)
        if (info.st_dev, info.st_ino) != expected[target.root]:
            raise TargetChangedError("Workspace root changed")
        current = target.root
        for component in target.path.relative_to(target.root).parts[:-1]:
            current = current / component
            next_fd = os.open(component, flags, dir_fd=parent_fd)
            info = os.fstat(next_fd)
            if (info.st_dev, info.st_ino) != expected.get(current):
                os.close(next_fd)
                raise TargetChangedError("Parent directory changed")
            os.close(parent_fd)
            parent_fd = next_fd
        info = os.stat(target.path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not same_signature(signature(info), target.file_signature):
            raise TargetChangedError("File changed before removal")
        recovery_id = uuid.uuid4().hex
        trash_fd = os.open(trash_dir, flags)
        os.mkdir(recovery_id, mode=0o700, dir_fd=trash_fd)
        backup_fd = os.open(recovery_id, flags, dir_fd=trash_fd)
        manifest = json.dumps({"workspace": str(target.root),
                               "path": target.path.relative_to(target.root).as_posix()}, ensure_ascii=False).encode("utf-8")
        manifest_fd = os.open("info.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                              0o600, dir_fd=backup_fd)
        try:
            with os.fdopen(manifest_fd, "wb", closefd=False) as stream:
                stream.write(manifest)
        finally:
            os.close(manifest_fd)
        os.rename(target.path.name, "file", src_dir_fd=parent_fd, dst_dir_fd=backup_fd)
        moved = os.stat("file", dir_fd=backup_fd, follow_symlinks=False)
        # Rename may change ctime; identity, size, mtime and link count must still match.
        observed = signature(moved)
        descriptor = os.open("file", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=backup_fd)
        try:
            moved_digest = file_digest(descriptor)
        finally:
            os.close(descriptor)
        if not same_signature(observed, target.file_signature) or moved_digest != target.file_digest:
            try:
                os.link("file", target.path.name, src_dir_fd=backup_fd, dst_dir_fd=parent_fd,
                        follow_symlinks=False)
            except OSError:
                pass  # Preserve the moved entry in recovery storage without overwriting a new file.
            raise TargetChangedError("File changed during removal; recovery copy preserved")
        return recovery_id
    finally:
        if backup_fd is not None:
            os.close(backup_fd)
        if trash_fd is not None:
            os.close(trash_fd)
        os.close(parent_fd)
