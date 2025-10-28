#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.


"""
将 docs/*.md 的首个标题与文件名对齐：
- 规则：标题 = 去除时间戳前缀后的文件名（去扩展名）。
- 仅处理 docs 目录下的 Markdown。
- 若文件无标题，则在文件最前插入一行 "# <标题>"，并保留原内容。
- 编码：UTF-8（无 BOM），统一 LF 行尾。

用法：
  python scripts/ensure_title_equals_filename.py [<files_or_dirs>...]
  - 未提供参数时，默认遍历 docs 目录下全部 .md
  - 传入目录时，递归处理其中的 .md 文件
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

TS_PREFIX_RE = re.compile(r"^(\d+)_([\s\S]+)\.md$", re.IGNORECASE)


def collect_targets(args: List[str]) -> List[Path]:
    paths: List[Path] = []
    if not args:
        if DOCS.is_dir():
            paths = [p for p in DOCS.rglob("*.md") if p.is_file()]
        return sorted(paths)

    for a in args:
        p = (ROOT / a).resolve() if not os.path.isabs(a) else Path(a)
        if p.is_dir():
            paths.extend([x for x in p.rglob("*.md") if x.is_file()])
        elif p.is_file() and p.suffix.lower() == ".md":
            paths.append(p)
    # 仅保留 docs 下的文件
    docs_paths = []
    try:
        docs_root = DOCS.resolve()
    except Exception:
        docs_root = DOCS
    for p in paths:
        try:
            if docs_root in p.resolve().parents or p.resolve() == docs_root:
                docs_paths.append(p)
        except Exception:
            # 若解析失败，保守跳过
            pass
    return sorted(set(docs_paths))


def expected_title_for(path: Path) -> str:
    name = path.name
    m = TS_PREFIX_RE.match(name)
    if m:
        return m.group(2)
    return path.stem


def read_text(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    # 兼容 CRLF/LF
    nl = "\r\n" if b"\r\n" in data else "\n"
    return data.decode("utf-8-sig", errors="replace"), nl


def write_text(path: Path, text: str, nl: str) -> None:
    # 统一写回 UTF-8（无 BOM）+ LF；与 .gitattributes 约定一致
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def ensure_title_equals_filename(path: Path) -> bool:
    try:
        text, nl = read_text(path)
    except Exception:
        return False
    lines = text.splitlines()
    # 查找首个标题行（# 开头）
    idx = None
    for i, ln in enumerate(lines[:20]):
        if ln.lstrip().startswith("#"):
            idx = i
            break
    title = expected_title_for(path)
    expected = f"# {title}"
    changed = False
    if idx is None:
        lines = [expected, ""] + lines
        changed = True
    else:
        if lines[idx] != expected:
            lines[idx] = expected
            changed = True
    if changed:
        write_text(path, "\n".join(lines), nl)
    return changed


def main(argv: List[str]) -> int:
    targets = collect_targets(argv)
    updated = 0
    for p in targets:
        try:
            if ensure_title_equals_filename(p):
                updated += 1
                print(f"[ensure_title] updated: {p.relative_to(ROOT)}")
        except Exception as e:
            print(f"[ensure_title] skip {p}: {e}")
    print(f"[ensure_title] processed={len(targets)} updated={updated}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
