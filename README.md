# G008

This repository hosts scaffolding for a Soft Actor-Critic (SAC) implementation located in `src/rl_sac`. The current code base provides class and function skeletons that describe the flow of data between replay buffers, policy/value networks, and training routines. A lightweight training demonstration is available to show how the components fit together when driven by synthetic data distilled from the bundled example article.

## 开发协议（Development Protocol）

* 演示脚本将策略网络视为**微型 LLM 头部**，直接读取“上一轮摘要 + 当前章节全文”的拼接文本并生成新的摘要。
* `data/sample_article.txt` 使用 `"[----------------------------------------------------->"` 作为段落分割符号，模拟教师模型输出来的分段提示。
* 训练过程中对每个分割执行**迭代摘要**：第 1 个摘要默认为空字符串，将其与第 1 个分割（两个分隔符之间的内容）拼接后得到第 1 次输出；随后把该摘要与第 2 个分割组合生成第 2 次输出，如此迭代，模拟蒸馏时“上一次摘要 + 间隔内容 → 新摘要”的累积推理轨迹。环境不会裁剪策略给出的文本，奖励函数只根据长度偏差、覆盖率与复制率进行评分。
* 开发前请先在当前环境中安装 `numpy` 与 `pytorch`（可直接运行 `scripts/install_pytorch.sh`，该脚本会顺带安装 `numpy`）。

## Examples

The `data/` directory contains sample textual material that mimics the structure of articles used throughout the project. For instance, `data/sample_article.txt` 提供了一篇多段落的中文示例文章，围绕状态表示、策略参数化以及评估流程等 SAC 概念展开，并补充了离线数据融合、超参数搜索与未来展望等段落。这些文字被刻意写得较长，以便验证分片处理与批量载入逻辑。文件通过 `"[----------------------------------------------------->"` 分隔段落，从而便于下游工具将其视作教师模型输出的逐段提示。

### Loading the sample article

You can load the example document using standard Python file operations. The snippet below demonstrates how to stream the file and split it into paragraphs for further preprocessing:

```python
from pathlib import Path

example_path = Path("data/sample_article.txt")
text = example_path.read_text(encoding="utf-8")
intervals = [
    interval.strip()
    for interval in text.split("[----------------------------------------------------->")
    if interval.strip()
]

for idx, interval in enumerate(intervals, start=1):
    print(f"Interval {idx}: {interval[:60]}...")
```

This workflow mirrors the intended usage within data ingestion pipelines, ensuring that each section of the article can be independently tokenized or transformed before feeding into SAC-related training tasks.

### Inspecting chapter previews and copy metrics

The demo now works纯文本输入，可以利用 `src.train_demo.analyze_summary` 检查摘要与章节之间的复制程度、覆盖率和长度比例。示例脚本如下：

```python
from pathlib import Path

DELIMITER = "[----------------------------------------------------->"
article = Path("data/sample_article.txt").read_text(encoding="utf-8")
chapters = [chunk.strip() for chunk in article.split(DELIMITER) if chunk.strip()]

from src.train_demo import analyze_summary, _format_text_debug

for index, chapter in enumerate(chapters, start=1):
    chars, preview = _format_text_debug(chapter, head=30, tail=30)
    metrics = analyze_summary("", chapter)
    print(
        f"Chapter {index:02d} | chars={chars:04d} copy≈{metrics['copy_ratio']:.2f} "
        f"coverage≈{metrics['coverage_ratio']:.2f} preview=\"{preview}\""
    )
```

这些信息与训练日志一致：每次 step 都会打印前后各 20 个字符的预览，以及以章节全文为基准计算的长度比例与复制率。摘要完全由策略网络生成，环境不会再按固定上限截断文本，惩罚项仅在出现大段抄写时发挥作用。

## Demo training run

The repository ships with a `train_demo.py` module under `src/` that wires together the replay buffer, agent, and trainer scaffolding using a toy environment constructed from the sample article statistics and iterative distillation summaries.

### Dependencies

The demo requires Python 3.10+ and the CPU build of [PyTorch](https://pytorch.org/). Optionally create and activate a virtual environment before installing the dependencies and running the script:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
scripts/install_pytorch.sh
```

> 若不希望创建虚拟环境，也可以直接执行 `scripts/install_pytorch.sh`，脚本会升级 `pip` 并安装 CPU 版本的 PyTorch（使用官方 `https://download.pytorch.org/whl/cpu` 镜像）。

### Running the demo

Execute the module from the repository root. Ensure `src/` is available on `PYTHONPATH` (for example by activating the virtual environment above) and run it with `-m`:

```bash
PYTHONPATH=src python -m train_demo --rounds 3
# or, thanks to the `src/__init__.py` package initializer:
python -m src.train_demo --rounds 3
```

每轮训练固定遍历 `data/sample_article.txt` 的全部 76 个分割片段，因此每个迭代（iteration）恰好对应一次环境 step，`--rounds` 仅控制重复轮次（默认 1000 轮）。脚本会在完成 76 个交互后集中执行一批 SAC 更新，数量与步骤数一致，从而模拟“先收集一整轮经验，再统一回放训练”的节奏。需要缩减或扩充集中训练的强度时，可以通过 `--post-round-updates` 覆盖默认值；`--replay-capacity` 则依旧决定演示缓冲区能保留多少过往转换。针对快速冒烟测试，还可以附加 `--max-chapters 2`（或任意正整数）限制每轮使用的章节数量，从而在几次 step 内观察完整的日志与训练流程。

为避免模型简单地逐字抄写原文，环境奖励会鼓励摘要长度靠近章节长度的 20%，并在检测到较长公共子串时施加惩罚；复制率、覆盖率与长度比例均会在日志中打印，便于观察策略是否保持压缩与去重能力。

### Expected output

The command prints a short training log summarizing the reward, replay buffer size, placeholder policy loss, and the copy diagnostics (length ratio, copy ratio, coverage) for each simulated step. Example output:

```
Loaded article debug info: chars=12345 preview="示例文本...结尾片段"
Chapter 01 | tokens≈0123 chars=0456 preview="段落起始...段落末尾"
...
Configured schedule: steps_per_round=76 post_round_updates=76
=== Training round 1 | steps=76 ===
  Step 01 | prev_summary=0000 chars ""
           | chapter=0456 chars "段落起始...段落末尾"
           -> summary=0098 chars "策略输出前缀...策略输出后缀" length_ratio=0.22 copy_ratio=0.18 reward=-0.41
...
    Update 076 | policy_loss=-0.1234 q1_loss=0.5678 q2_loss=0.9123 avg_reward=-0.4321
    Post-round metric averages | policy_loss=-0.2345 q1_loss=0.4567 q2_loss=0.8910 average_reward=-0.3210
```

Actual numbers vary because the demo samples synthetic actions stochastically, but the structure of the log should match the example. Each step reports both the character length and a head/tail preview of the current input segment, while the iterative summary preview reflects outputs padded to at least 20% of the accumulated input tokens. After 76 steps finish, the trainer prints一个集中更新阶段的详情：逐次的策略/价值损失以及整轮的平均指标，帮助观察批量回放的收敛趋势。

### Saved artifacts

After the log finishes, the script 序列化一个模型快照到 `out/demo_agent_snapshot.json`，其中包含演示代理的占位符状态与运行元数据（如训练步数、经验回放容量）。该代理始终在 CPU 上训练，并记录策略头部的参数数量，同时标注导出的模型体积。为了满足新的存档协议，脚本会在 `out/demo_agent_model.bin` 写出一个精确 199 MB（209,460,851 字节）的二进制模型文件，用以模拟重量级微型 LLM 头部的交付物。所有产物会自动创建父目录 `out/`，便于在多阶段流程中复用或进一步加工演示产出的检查点。
