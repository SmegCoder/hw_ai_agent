from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


class TargetChangedError(OSError):
    pass


def signature(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink)


@dataclass(frozen=True)
class WriteTarget:
    root: Path
    path: Path
    original: str
    file_signature: tuple[int, ...] | None
    directories: tuple[tuple[Path, int, int], ...]


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
    else:
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TargetChangedError("Target must be a regular file without hard links")
        file_signature = signature(info)
    return WriteTarget(root, path, original, file_signature, tuple(directories))


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
            if target.file_signature is None or signature(info) != target.file_signature:
                raise TargetChangedError("The approved file changed")
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, TargetChangedError):
            raise
        raise TargetChangedError("Cannot verify the approved target") from None


def write_target(target: WriteTarget, content: str) -> None:
    data = content.encode("utf-8")
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
        flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if target.file_signature is None:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            file_fd = os.open(target.path.name, flags, 0o666, dir_fd=directory_fd)
        except OSError:
            raise TargetChangedError("Cannot open the same approved target safely") from None
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TargetChangedError("Opened target is not a regular, unlinked file")
        if target.file_signature is not None and signature(info) != target.file_signature:
            raise TargetChangedError("File replaced or edited during execution")
        if target.file_signature is not None:
            os.ftruncate(file_fd, 0)
        view = memoryview(data)
        while view:
            count = os.write(file_fd, view)
            view = view[count:]
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)
