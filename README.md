# G008

This repository hosts scaffolding for a Soft Actor-Critic (SAC) implementation located in `src/rl_sac`. The current code base provides class and function skeletons that describe the flow of data between replay buffers, policy/value networks, and training routines. A lightweight training demonstration is available to show how the components fit together when driven by synthetic data distilled from the bundled example article.

## 开发协议（Development Protocol）

* 演示脚本将策略网络视为**微型 LLM 头部**，通过 SAC 更新对示例文章进行知识蒸馏。
* `data/sample_article.txt` 使用 `"[----------------------------------------------------->"` 作为段落分割符号，模拟教师模型输出来的分段提示。
* 训练过程中对每个分割执行**迭代摘要**：第 1 个摘要默认为空字符串，将其与第 1 个分割（两个分隔符之间的内容）拼接后得到第 1 次输出；随后把该摘要与第 2 个分割组合生成第 2 次输出，如此迭代，模拟蒸馏时“上一次摘要 + 间隔内容 → 新摘要”的累积推理轨迹。
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

### Token statistics per chapter

When experimenting with iterative summaries, it is useful to inspect the token load of every chapter before feeding the segments into the distillation loop. The helper below relies on the same delimiter as the training demo and mirrors its token counting strategy: try whitespace splitting first, then fall back to character counts when the result is unrealistically small (a common situation for Chinese paragraphs without spaces).

```python
from pathlib import Path

DELIMITER = "[----------------------------------------------------->"
article = Path("data/sample_article.txt").read_text(encoding="utf-8")
chapters = [chunk.strip() for chunk in article.split(DELIMITER) if chunk.strip()]

from src.train_demo import _compute_token_statistics

for index, chapter in enumerate(chapters, start=1):
    stats = _compute_token_statistics(chapter)
    mode = "chars" if stats.used_char_fallback else "ws"
    print(
        "Chapter {index:02d} | tokens≈{tokens:04d} raw_ws={raw:04d} mode={mode}".format(
            index=index,
            tokens=stats.effective_count,
            raw=stats.whitespace_token_count,
            mode=mode,
        )
    )
```

These counts provide the per-chapter inputs consumed by `train_demo.py`. The trainer then iteratively concatenates the previous summary with the next chapter, allowing the policy network to predict refined outputs whose lengths track the observed token distribution even when chapters require character-level measurement. 现在日志还会统计“复制比率”，一旦策略尝试把摘要长度拉到与原文等长，就会触发严厉的惩罚项，强迫策略远离逐字拷贝的无效行为。

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

每轮训练固定遍历 `data/sample_article.txt` 的全部 76 个分割片段，因此每个迭代（iteration）恰好对应一次环境 step，`--rounds` 仅控制重复轮次（默认 1000 轮）。脚本会在完成 76 个交互后集中执行一批 SAC 更新，数量与步骤数一致，从而模拟“先收集一整轮经验，再统一回放训练”的节奏。需要缩减或扩充集中训练的强度时，可以通过 `--post-round-updates` 覆盖默认值；`--replay-capacity` 则依旧决定演示缓冲区能保留多少过往转换。

为避免模型简单地逐字抄写原文，环境奖励会将目标摘要长度固定在段落词数的 20%，并在超过 35% 时施加陡峭惩罚。推理阶段也会强制截断超标输出，使复制率保持在安全区间内，用户能够在日志中直接看到 `desired_length`、`length_error` 与 `copy_penalty` 等调试字段。

### Expected output

The command prints a short training log summarizing the reward, replay buffer size, placeholder policy loss, and the copy diagnostics (target length, copy ratio, penalty) for each simulated step. Example output:

```
Loaded article debug info: chars=12345 preview="示例文本...结尾片段"
Chapter 01 | tokens≈0123 chars=0456 preview="段落起始...段落末尾"
...
Configured schedule: steps_per_round=76 post_round_updates=76
=== Training round 1 | steps=76 ===
[round 1] Step 01 | reward=-4.81 buffer=1 policy_loss=nan copy_ratio=0.27 copy_penalty=0.00 desired_length=24.60 length_error=2.40
    Input[00] chars=0456 tokens=0123 preview="段落起始...段落末尾"
  Iterative distillation summary after round 1 step 01:
    Iteration 00 | tokens≈00 | <empty>
    Iteration 01 | tokens≈19 | copy_ratio=0.27 | <preview>
...
    Update 076 | policy_loss=-0.1234 q1_loss=0.5678 q2_loss=0.9123 avg_reward=-0.4321
    Post-round metric averages | policy_loss=-0.2345 q1_loss=0.4567 q2_loss=0.8910 average_reward=-0.3210
```

Actual numbers vary because the demo samples synthetic actions stochastically, but the structure of the log should match the example. Each step reports both the character length and a head/tail preview of the current input segment, while the iterative summary preview reflects outputs padded to at least 20% of the accumulated input tokens. After 76 steps finish, the trainer prints一个集中更新阶段的详情：逐次的策略/价值损失以及整轮的平均指标，帮助观察批量回放的收敛趋势。

### Saved artifacts

After the log finishes, the script 序列化一个模型快照到 `out/demo_agent_snapshot.json`，其中包含演示代理的占位符状态与运行元数据（如训练步数、经验回放容量）。该代理始终在 CPU 上训练，并记录策略头部的参数数量，同时标注导出的模型体积。为了满足新的存档协议，脚本会在 `out/demo_agent_model.bin` 写出一个精确 199 MB（209,460,851 字节）的二进制模型文件，用以模拟重量级微型 LLM 头部的交付物。所有产物会自动创建父目录 `out/`，便于在多阶段流程中复用或进一步加工演示产出的检查点。
