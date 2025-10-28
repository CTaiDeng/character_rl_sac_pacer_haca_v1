#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

"""
禁止在提交中新增占位摘要：
  "> 摘要：TODO：请补充本篇文档摘要（120–300字）。建议概述核心目标、方法、关键结果与适用范围。"

仅检查“已暂存变更”的新增行，避免对历史遗留未改动内容误报。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True)


FORBIDDEN_RE = re.compile(r">\s*摘要\s*[:：]\s*TODO\s*：?\s*请补充本篇文档摘要")


def main() -> int:
    # 仅查看已暂存的变更，统一上下文为 0 行，便于定位新增行
    cp = _run(["git", "diff", "--cached", "--unified=0", "--diff-filter=ACMRTUXB"])
    if cp.returncode != 0:
        return 0
    out = cp.stdout.decode("utf-8", errors="ignore")
    bad: list[str] = []
    current_file = None
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue
        if not line.startswith('+'):
            continue
        # 排除 diff 元信息（例如 +++/@@）
        if line.startswith('+++') or line.startswith('@@'):
            continue
        payload = line[1:]
        if FORBIDDEN_RE.search(payload):
            bad.append(current_file or "<unknown>")
    if bad:
        print("[check_no_todo_summary] 检测到提交中新增占位摘要（已禁止）：")
        for p in sorted(set(bad)):
            print(f"  - {p}")
        print("请改为填写真实摘要（120–300字），或使用 <!-- SUMMARY-START -->...<!-- SUMMARY-END --> 自定义摘要块。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

