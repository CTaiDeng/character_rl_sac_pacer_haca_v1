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
文档对齐指令（安全模式）：默认不修改 docs 知识库，仅对 README 执行所需更新。

默认行为（不修改 docs/*）：
- 重建 README 文末的“文档摘要索引”（仅写 README）；
- 修复 README 索引中的样式问题；
- 将 README 中遗留的 $\texttt{...}$ 规范为反引号 `...`；
- 对 README 执行 Markdown 规范化（UTF-8 无 BOM + LF，数学分隔统一）。

可选行为（需要显式开启）：
- 通过传参 `--mutate-docs` 才会执行以下对 docs/* 的改动：
  1) 重写 docs/<ts>_*.md 的时间戳前缀为 git 首次入库时间；
  2) 在文档主标题下写入/更新“日期：YYYY-MM-DD”；
  3) 对 docs 执行 Markdown 规范化；
  4) 对 docs 下 Markdown 执行 $\texttt{...}$ → 反引号 的转换。

用法：
  python scripts/align_docs.py               # 仅更新 README，不改 docs
  python scripts/align_docs.py --mutate-docs # 同步更新 docs（可能改名/写入）
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str]) -> int:
    try:
        print("[align_docs] $", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(ROOT))
    except Exception as e:
        print(f"[align_docs] 运行失败: {e}")
        return 1


def main() -> int:
    rc = 0
    mutate_docs = ('--mutate-docs' in sys.argv)

    # 仅当允许修改 docs 时，执行对 docs 的更改
    if mutate_docs:
        rc |= run([sys.executable, str(ROOT / 'scripts' / 'rename_docs_to_git_ts.py')])
        rc |= run([sys.executable, str(ROOT / 'scripts' / 'insert_doc_date_from_prefix.py')])

    # 无论是否修改 docs，都更新 README 的索引（只写 README，不触碰 docs 内容）
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'update_readme_index.py')])

    # 清理 README 索引中的样式问题（只写 README）
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'fix_readme_index_style.py')])

    # 仅对 README 进行 \texttt → 反引号 的转换，避免修改 docs
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'convert_texttt_to_backticks.py'), 'README.md'])

    # 规范化 README；如明确允许修改 docs，再规范化 docs
    rc |= run([sys.executable, str(ROOT / 'scripts' / 'md_normalize.py'), 'README.md'])
    if mutate_docs:
        rc |= run([sys.executable, str(ROOT / 'scripts' / 'md_normalize.py'), 'docs'])

    if rc == 0:
        print('[align_docs] 文档对齐完成（docs 未修改）' if not mutate_docs else '[align_docs] 文档对齐完成（包含 docs 更改）')
    else:
        print('[align_docs] 文档对齐存在错误，请查看上方输出')
    return rc


if __name__ == '__main__':
    sys.exit(main())
