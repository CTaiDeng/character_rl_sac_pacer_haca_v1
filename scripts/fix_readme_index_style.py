#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 README.md 文末“文档摘要索引”块内的反引号文件路径改为 $\texttt{...}$，以符合 Markdown 规范。
仅替换 START/END 标记之间、以 "- `docs/" 开头的行；不改动块标题与摘要文本。
"""

import os

START_MARK = "<!-- DOCS-SUMMARY-INDEX:START -->"
END_MARK = "<!-- DOCS-SUMMARY-INDEX:END -->"


def read_text(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    nl = '\r\n' if b'\r\n' in data else '\n'
    return data.decode('utf-8-sig', errors='replace'), nl


def write_text(path: str, text: str, nl: str):
    # 统一 UTF-8（无 BOM）+ LF
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    readme = os.path.join(root, 'README.md')
    text, nl = read_text(readme)
    s = text.find(START_MARK)
    e = text.find(END_MARK)
    if s < 0 or e < 0 or e <= s:
        print('[fix_readme_index_style] 未找到索引块，跳过')
        return
    head = text[:s]
    block = text[s:e]
    tail = text[e:]
    lines = block.splitlines()
    changed = False
    def _escape_texttt(s: str) -> str:
        return s.replace('_', r'\_')

    for i, ln in enumerate(lines):
        t = ln.lstrip()
        if t.startswith('- $\\texttt{') and t.endswith('}$'):
            # 将 $\texttt{...}$ 还原为反引号，并恢复下划线
            inner = t[len('- $\\texttt{'):-2]
            inner_unescaped = inner.replace('\\_', '_')
            new_ln = ln[: len(ln) - len(t)] + f"- `{inner_unescaped}`"
            if new_ln != ln:
                lines[i] = new_ln
                changed = True
    if not changed:
        print('[fix_readme_index_style] 索引块内无需更改')
        return
    new_block = '\n'.join(lines)
    new_text = head + new_block + tail
    write_text(readme, new_text, nl)
    print('[fix_readme_index_style] 已将索引条目改为反引号行内代码形式')


if __name__ == '__main__':
    main()
