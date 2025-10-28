#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
根据 .gitattributes 自动修复“已暂存”的文本文件：
- 目标：仅处理被标记为 eol=lf 的文件；
- 行为：移除 UTF-8 BOM（EF BB BF），并将 CRLF/CR 统一为 LF；
- 若发生改动，写回工作区文件并重新加入暂存区（git add）。

注意：本脚本用于预提交钩子中的“纠错”步骤，随后仍应由
scripts/check_eol_utf8_by_gitattributes.py 进行严格校验。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True)


def _staged_files() -> list[str]:
    cp = _run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"])
    if cp.returncode != 0:
        return []
    return [line.strip() for line in cp.stdout.decode("utf-8", errors="ignore").splitlines() if line.strip()]


def _attr_eol(path: str) -> str | None:
    cp = _run(["git", "check-attr", "eol", "--", path])
    if cp.returncode != 0:
        return None
    out = cp.stdout.decode("utf-8", errors="ignore").strip()
    parts = out.split(":", 2)
    if len(parts) >= 3:
        val = parts[-1].strip()
        return val or None
    return None


def _read_worktree_bytes(path: str) -> bytes | None:
    p = ROOT / path
    if not p.exists() or not p.is_file():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def _write_worktree_bytes(path: str, data: bytes) -> bool:
    p = ROOT / path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return True
    except Exception:
        return False


_re_crlf = re.compile(br"\r\n")
_re_cr_solo = re.compile(br"\r(?!\n)")


def _fix_bytes(data: bytes) -> tuple[bytes, bool, int, int]:
    changed = False
    crlf = len(_re_crlf.findall(data))
    crs = len(_re_cr_solo.findall(data))
    if data.startswith(b"\xEF\xBB\xBF"):
        data = data[3:]
        changed = True
    if crlf or crs:
        data = _re_crlf.sub(b"\n", data)
        data = _re_cr_solo.sub(b"\n", data)
        changed = True
    return data, changed, crlf, crs


def main() -> int:
    files = _staged_files()
    if not files:
        return 0

    fixed = 0
    for path in files:
        eol = _attr_eol(path)
        if not eol or eol.lower() != "lf":
            continue
        wt = _read_worktree_bytes(path)
        if wt is None:
            continue
        new_wt, changed_wt, crlf, crs = _fix_bytes(wt)
        # 如果工作区需要修复，则写回并加入暂存区
        if changed_wt:
            try:
                new_wt.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                # 保守：不自动更改非 UTF-8 文本
                pass
            else:
                if _write_worktree_bytes(path, new_wt):
                    cp = _run(["git", "add", "--", path])
                    if cp.returncode == 0:
                        fixed += 1
            continue
        # 若工作区已是理想状态，但暂存区仍含 BOM/CRLF，则强制刷新暂存区
        cp_blob = _run(["git", "show", f":{path}"])
        if cp_blob.returncode != 0:
            continue
        idx = cp_blob.stdout
        if idx.startswith(b"\xEF\xBB\xBF") or (b"\r\n" in idx):
            # 工作区无 BOM/LF，直接重新 add 以覆盖暂存区
            cp = _run(["git", "add", "--", path])
            if cp.returncode == 0:
                fixed += 1

    if fixed:
        print(f"[fix_staged_eol_utf8] 已自动修复并更新暂存区：{fixed} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
