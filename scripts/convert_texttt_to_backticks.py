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
将仓库内 Markdown 文档中的 $\texttt{...}$ 行内片段批量转换为反引号 `...`，
仅在代码围栏外进行替换：
  - 匹配形如 $\texttt{a\_b}$ 的片段；
  - 将内容中的 \_ 还原为 _；
  - 输出为 `a_b`。

用法：
  python scripts/convert_texttt_to_backticks.py            # 全仓 .md
  python scripts/convert_texttt_to_backticks.py a.md b.md  # 指定文件
"""

import os
import re
import sys
from typing import List, Tuple

FENCE_RE = re.compile(r"^\s*(```|~~~)")
TEXTTT_RE = re.compile(r"\$\s*\\texttt\{([^}]+)\}\s*\$")


def read_text(path: str) -> Tuple[str, str]:
    with open(path, 'rb') as f:
        data = f.read()
    nl = '\r\n' if b'\r\n' in data else '\n'
    text = data.decode('utf-8-sig', errors='replace')
    return text, nl


def write_text(path: str, text: str, nl: str) -> None:
    # 统一 UTF-8（无 BOM）+ LF
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def split_by_fences(text: str):
    lines = text.splitlines(keepends=True)
    segments = []
    buf = []
    in_code = False
    for ln in lines:
        if FENCE_RE.match(ln):
            if buf:
                segments.append((in_code, ''.join(buf)))
                buf = []
            segments.append((True, ln))
            in_code = not in_code
        else:
            buf.append(ln)
    if buf:
        segments.append((in_code, ''.join(buf)))
    return segments


def convert_segment(seg: str) -> str:
    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace(r"\_", "_")
        return f"`{inner}`"

    return TEXTTT_RE.sub(_repl, seg)


def process_file(path: str) -> bool:
    text, nl = read_text(path)
    parts = split_by_fences(text)
    changed = False
    out_parts: List[str] = []
    for in_code, seg in parts:
        if in_code:
            out_parts.append(seg)
        else:
            new_seg = convert_segment(seg)
            if new_seg != seg:
                changed = True
            out_parts.append(new_seg)
    if changed:
        write_text(path, ''.join(out_parts), nl)
    return changed


def find_all_md(root: str) -> List[str]:
    out: List[str] = []
    skip = {'.git', '.idea', 'out', 'dist', 'build', '__pycache__'}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip]
        for fn in fns:
            if fn.lower().endswith('.md'):
                out.append(os.path.join(dp, fn))
    return out


def main(argv: List[str]) -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    files = [os.path.abspath(p) for p in argv] if argv else find_all_md(root)
    changed = 0
    for p in files:
        if not os.path.isfile(p):
            continue
        try:
            if process_file(p):
                changed += 1
        except Exception as e:
            print(f"[convert_texttt_to_backticks] 跳过 {p}: {e}")
    print(f"[convert_texttt_to_backticks] 已转换 {changed} 个 Markdown 文件")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
