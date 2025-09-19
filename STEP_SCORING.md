# 当前 Step 打分方案说明

## 文字描述

在每个迭代 step 中，环境会拿到策略生成的摘要文本 $s$ 与对应章节原文 $c$，并基于二者的字符级匹配关系即时计算奖励。奖励不会依赖长度目标或截断规则，而是通过相似度、覆盖率与新颖度综合评估摘要质量：

- **相似度 $\mathrm{sim}(s, c)$**：使用 `difflib.SequenceMatcher` 的 `ratio()` 作为字符序列的全局相似度。
- **覆盖率 $\mathrm{cov}(s, c)$**：统计匹配块的字符总数与章节总字符数之比，衡量摘要对原文信息的涵盖程度。
- **复制率 $\mathrm{copy}(s, c)$**：取最长匹配块长度与摘要总长度之比，表示摘要中最大连续片段对原文的直接复制程度；新颖度定义为 $\mathrm{nov}(s, c) = \max(0, 1 - \mathrm{copy}(s, c))$。

最终奖励 $R(s, c)$ 是三项指标的加权和：
\[
R(s, c) = 0.6 \cdot \mathrm{sim}(s, c) + 0.3 \cdot \mathrm{cov}(s, c) + 0.1 \cdot \mathrm{nov}(s, c).
\]

权重反映了我们优先追求整体语义相似和信息覆盖，同时对保持一定的新颖度给予次要奖励。

## 伪代码

```pseudo
function compute_step_reward(summary_text, chapter_text):
    matcher = SequenceMatcher(summary_text, chapter_text)
    similarity = matcher.ratio()

    match_blocks = matcher.get_matching_blocks()
    matched_chars = sum(block.size for block in match_blocks)
    longest_block = max(block.size for block in match_blocks, default=0)

    chapter_len = length(chapter_text)
    summary_len = length(summary_text)

    if chapter_len == 0:
        coverage = 0
    else:
        coverage = matched_chars / chapter_len

    if summary_len == 0:
        copy_ratio = 0
    else:
        copy_ratio = longest_block / summary_len

    novelty = max(0, 1 - copy_ratio)

    reward = 0.6 * similarity + 0.3 * coverage + 0.1 * novelty
    return reward, {
        "similarity": similarity,
        "coverage_ratio": coverage,
        "copy_ratio": copy_ratio,
        "novelty_ratio": novelty
    }
```

## 更新约定

若未来调整任一权重或改动指标的定义（例如改用其他相似度算法、覆盖率统计方式等），必须同步修改本文件中相应的数学公式、文字描述与伪代码，确保文档与实现保持一致。
