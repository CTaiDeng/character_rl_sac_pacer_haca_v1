#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 .gitattributes 校验暂存区中文件是否满足：UTF-8（无 BOM）+ LF。

规则：
- 仅检查在 index 中已暂存的文件（diff --cached）；
- 仅检查 .gitattributes 解析为 eol=lf 的文本文件；
- 对于每个文件，读取暂存版本（git show :path）验证：
  1) 不以 UTF-8 BOM (EF BB BF) 开头；
  2) 字节可按 UTF-8 严格解码；
  3) 不包含 CRLF（\r\n）。

若发现问题，打印文件与原因并以非零状态退出以阻止提交。
"""

from __future__ import annotations

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
    # Format: "<path>: eol: <value>"; value may be "lf", "crlf", or "unspecified"
    parts = out.split(":", 2)
    if len(parts) >= 3:
        return parts[-1].strip() or None
    return None


def _read_index_bytes(path: str) -> bytes | None:
    cp = _run(["git", "show", f":{path}"])
    if cp.returncode != 0:
        return None
    return cp.stdout


def main() -> int:
    files = _staged_files()
    if not files:
        return 0
    errors: list[tuple[str, str]] = []
    for path in files:
        eol = _attr_eol(path)
        if eol is None:
            continue
        if eol.lower() != "lf":
            # Skip files configured as CRLF or unspecified
            continue
        data = _read_index_bytes(path)
        if data is None:
            continue
        if data.startswith(b"\xEF\xBB\xBF"):
            errors.append((path, "包含 UTF-8 BOM (EF BB BF)"))
            continue
        try:
            data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            errors.append((path, "不是严格的 UTF-8 可解码文本"))
            continue
        if b"\r\n" in data:
            errors.append((path, "检测到 CRLF 行尾，期望 LF"))
            continue
    if errors:
        print("[check_eol_utf8_by_gitattributes] 以下文件不符合 UTF-8（无 BOM）+ LF 要求：")
        for p, reason in errors:
            print(f"  - {p}: {reason}")
        print("请移除 BOM、转换为 UTF-8 严格编码，并统一为 LF 行尾后重试提交。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
