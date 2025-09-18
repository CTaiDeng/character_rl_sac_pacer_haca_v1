# G008

This repository hosts scaffolding for a Soft Actor-Critic (SAC) implementation located in `src/rl_sac`. The current code base provides class and function skeletons that describe the flow of data between replay buffers, policy/value networks, and training routines. A lightweight training demonstration is available to show how the components fit together when driven by synthetic data derived from the bundled example article.

## Examples

The `examples/` directory contains sample textual material that mimics the structure of articles used throughout the project. For instance, `examples/sample_article.txt` 提供了一篇多段落的中文示例文章，围绕状态表示、策略参数化以及评估流程等 SAC 概念展开。These paragraphs are intended to be processed as independent chunks by downstream tooling.

### Loading the sample article

You can load the example document using standard Python file operations. The snippet below demonstrates how to stream the file and split it into paragraphs for further preprocessing:

```python
from pathlib import Path

example_path = Path("examples/sample_article.txt")
text = example_path.read_text(encoding="utf-8")
paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]

for idx, paragraph in enumerate(paragraphs, start=1):
    print(f"Paragraph {idx}: {paragraph[:60]}...")
```

This workflow mirrors the intended usage within data ingestion pipelines, ensuring that each section of the article can be independently tokenized or transformed before feeding into SAC-related training tasks.

## Demo training run

The repository ships with a `train_demo.py` module under `src/` that wires together the replay buffer, agent, and trainer scaffolding using a toy environment constructed from the sample article statistics.

### Dependencies

The demo only requires a Python interpreter (3.10 or newer) and does not depend on external libraries. Optionally create and activate a virtual environment before running the script:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### Running the demo

Execute the module from the repository root. Ensure `src/` is available on `PYTHONPATH` (for example by activating the virtual environment above) and run it with `-m`:

```bash
PYTHONPATH=src python -m train_demo --steps 8
# or, thanks to the `src/__init__.py` package initializer:
python -m src.train_demo --steps 8
```

The `--steps` flag controls the number of simulated environment interactions, while `--replay-capacity` adjusts the maximum number of transitions retained in the demo buffer.

### Expected output

The command prints a short training log summarizing the reward, replay buffer size, and placeholder policy loss for each simulated step. Example output:

```
Step 01 | reward=-10.54 buffer=1 policy_loss=nan
Step 02 | reward=-5.21 buffer=2 policy_loss=nan
Step 03 | reward=-4.97 buffer=3 policy_loss=nan
Step 04 | reward=-2.08 buffer=4 policy_loss=2.08
...
```

Actual numbers vary because the demo samples synthetic actions stochastically, but the structure of the log should match the example.
