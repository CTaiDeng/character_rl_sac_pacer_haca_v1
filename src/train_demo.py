"""Run a lightweight demonstration of the SAC scaffolding.

The script constructs a toy environment whose observations are
feature vectors derived from the paragraphs of
``data/sample_article.txt``. Minimal policy, value, replay buffer,
and trainer implementations are provided to exercise the public APIs of
``src/rl_sac`` without depending on deep learning frameworks.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, MutableMapping, Sequence

# Ensure the ``rl_sac`` package is importable when the module is executed
# without installing it as a distribution. This covers both ``python -m``
# execution (where ``src`` is on ``PYTHONPATH``) and direct invocation of the
# file path.
SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
OUT_DIR = REPO_ROOT / "out"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_sac.agent import AgentConfig, SACAgent
from rl_sac.networks import NetworkFactory, PolicyNetwork, QNetwork
from rl_sac.replay_buffer import BaseReplayBuffer, Transition
from rl_sac.trainer import Trainer, TrainerConfig


def load_article_features(path: Path) -> List[Sequence[float]]:
    """Load the sample article and convert paragraphs into feature vectors."""

    text = path.read_text(encoding="utf-8")
    paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    feature_vectors: List[Sequence[float]] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        tokens = paragraph.split()
        lengths = [len(token.strip(".,`")) for token in tokens]
        avg_token_length = statistics.fmean(lengths) if lengths else 0.0
        feature_vectors.append(
            (
                float(idx),
                float(len(tokens)),
                avg_token_length,
                float(sum(1 for token in tokens if token.isupper())),
            )
        )
    return feature_vectors


class ArticleEnvironment:
    """Deterministic environment backed by article paragraph statistics.

    Note: kept for backward-compatibility with the original immediate-reward
    demo. The new LongReadMDPEnvironment below models the delayed-reward MDP
    described in the issue statement.
    """

    def __init__(self, states: Sequence[Sequence[float]]) -> None:
        if not states:
            raise ValueError("The environment requires at least one state.")
        self._states = [tuple(map(float, state)) for state in states]
        self._cursor = 0

    def reset(self) -> Sequence[float]:
        self._cursor = 0
        return self._states[self._cursor]

    def step(self, action: Sequence[float]) -> Transition:
        state = self._states[self._cursor]
        next_index = (self._cursor + 1) % len(self._states)
        next_state = self._states[next_index]
        # Reward encourages the action to match the paragraph length feature.
        target_length = state[1]
        taken_length = action[0] if action else 0.0
        reward = -abs(target_length - taken_length)
        done = next_index == 0
        transition = Transition(
            state=state,
            action=tuple(map(float, action)),
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self._cursor = next_index
        return transition


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns 0.0 if any vector is all zeros to avoid division-by-zero.
    """

    ax = list(map(float, a))
    bx = list(map(float, b))
    if len(ax) != len(bx):
        raise ValueError("Vectors must have the same length for cosine similarity.")
    dot = sum(x * y for x, y in zip(ax, bx))
    na = sum(x * x for x in ax) ** 0.5
    nb = sum(y * y for y in bx) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class LongReadMDPEnvironment:
    """MDP for long-form reading with delayed terminal reward.

    State s_t = (summary_{t-1}, chunk_t), where each component is a 4-d vector
    derived from the article. The action a_t is the updated summary vector
    (summary_t). Intermediate rewards are 0; the final reward after the last
    chunk is the cosine similarity between the final summary and a simple
    ground-truth summary vector computed from the whole article.
    """

    def __init__(self, chunks: Sequence[Sequence[float]]) -> None:
        if not chunks:
            raise ValueError("The environment requires at least one chunk.")
        # Normalize chunk vectors to 4-D float tuples.
        self._chunks: List[tuple[float, float, float, float]] = [
            (float(c[0]), float(c[1]), float(c[2]), float(c[3])) for c in chunks
        ]
        # Ground-truth summary: mean of all chunk features (simple proxy).
        n = len(self._chunks)
        sums = [0.0, 0.0, 0.0, 0.0]
        for c in self._chunks:
            for i in range(4):
                sums[i] += c[i]
        self._target_summary = tuple(s / n for s in sums)
        self._cursor = 0
        self._summary: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def reset(self) -> Sequence[float]:
        self._cursor = 0
        self._summary = (0.0, 0.0, 0.0, 0.0)
        # Return concatenated state (summary, current chunk)
        chunk = self._chunks[self._cursor]
        return (*self._summary, *chunk)

    def step(self, action: Sequence[float]) -> Transition:
        # Parse action as the new summary vector
        action4 = tuple(float(x) for x in (list(action)[:4] if action else [0.0, 0.0, 0.0, 0.0]))
        state = (*self._summary, *self._chunks[self._cursor])

        # Apply action to update the summary
        self._summary = action4

        next_index = self._cursor + 1
        done = next_index >= len(self._chunks)
        if done:
            # Terminal reward: similarity to ground-truth summary
            reward = _cosine_similarity(self._summary, self._target_summary)
            # Next state wraps to the start for convenience
            next_state = (*self._summary, *self._chunks[0])
            next_index = 0
        else:
            reward = 0.0
            next_state = (*self._summary, *self._chunks[next_index])

        transition = Transition(
            state=tuple(map(float, state)),
            action=tuple(map(float, action4)),
            reward=float(reward),
            next_state=tuple(map(float, next_state)),
            done=done,
        )
        self._cursor = next_index
        return transition


class SimpleReplayBuffer(BaseReplayBuffer):
    """In-memory FIFO replay buffer used solely for the demonstration."""

    def add(self, transition: Transition) -> None:
        if len(self._storage) >= self._capacity:
            self._storage.pop(0)
        self._storage.append(transition)

    def sample(self, batch_size: int) -> Iterable[Transition]:
        if not self._storage:
            return []
        size = min(len(self._storage), batch_size)
        return random.sample(self._storage, size)


@dataclass
class DemoNetworkFactory(NetworkFactory):
    """Factory returning deterministic placeholder networks."""

    def build_policy(self, *args: Any, **kwargs: Any) -> "RandomPolicy":
        return RandomPolicy()

    def build_q_functions(self, *args: Any, **kwargs: Any) -> tuple["SimpleQNetwork", "SimpleQNetwork"]:
        return SimpleQNetwork(), SimpleQNetwork()


class RandomPolicy(PolicyNetwork):
    """Policy that generates actions from the provided state.

    Backward compatible with the original toy environment (1-D action) and the
    new long-read MDP (4-D summary update action).
    """

    def forward(self, state: Sequence[float]) -> tuple[List[float], MutableMapping[str, Any]]:
        info: MutableMapping[str, Any] = {}
        if len(state) >= 8:
            # State = (summary_prev[4], chunk[4])
            s_prev = list(state[:4])
            chunk = list(state[4:8])
            # Heuristic: move summary towards current chunk (simple fusion)
            fused = [(sp + ch) / 2.0 for sp, ch in zip(s_prev, chunk)]
            # Add small Gaussian noise for stochasticity
            noisy = [v + random.gauss(0.0, 0.1) for v in fused]
            info["mode"] = "mdp-4d"
            info["fused"] = fused
            return noisy, info
        else:
            # Original 1-D case: sample around paragraph length feature
            length_feature = state[1]
            action = [random.gauss(length_feature, 1.5)]
            info["mode"] = "legacy-1d"
            info["expected_length"] = length_feature
            return action, info

    def parameters(self) -> List[float]:  # pragma: no cover - placeholder
        return []


class SimpleQNetwork(QNetwork):
    """Q-network that scores actions.

    - For the long-read MDP (8-D state, 4-D action), returns negative L2
      distance between the proposed summary (action) and the current chunk.
    - For the legacy demo (4-D state, 1-D action), returns negative absolute
      error to the length feature.
    """

    def forward(self, state: Sequence[float], action: Sequence[float]) -> float:
        if len(state) >= 8:
            chunk = list(state[4:8])
            act = list((action or [])[:4])
            # pad if necessary
            while len(act) < 4:
                act.append(0.0)
            l2 = sum((a - c) * (a - c) for a, c in zip(act, chunk)) ** 0.5
            return -float(l2)
        else:
            target_length = state[1]
            taken_length = action[0] if action else 0.0
            return -abs(target_length - taken_length)

    def parameters(self) -> List[float]:  # pragma: no cover - placeholder
        return []


class DemoSACAgent(SACAgent):
    """Concrete SAC agent used for illustrating the training flow."""

    def __init__(self, *args: Any, update_batch_size: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update_batch_size = update_batch_size

    def act(self, state: Sequence[float], deterministic: bool = False) -> List[float]:
        # Support both legacy 1-D action and new 4-D summary update
        if len(state) >= 8:
            s_prev = list(state[:4])
            chunk = list(state[4:8])
            if deterministic:
                # Deterministic fusion
                action = [(sp + ch) / 2.0 for sp, ch in zip(s_prev, chunk)]
                return [float(x) for x in action]
            action, _ = self.policy.forward(state)
            # Ensure 4-D
            action4 = list((action or [])[:4])
            while len(action4) < 4:
                action4.append(0.0)
            return [float(x) for x in action4]
        else:
            if deterministic:
                # Match the paragraph length feature when acting deterministically.
                return [float(state[1])]
            action, _ = self.policy.forward(state)
            return [float(action[0])]

    def update(self) -> MutableMapping[str, float]:
        if len(self.replay_buffer) == 0:
            return {"policy_loss": 0.0, "q_loss": 0.0}
        batch = list(self.replay_buffer.sample(self.update_batch_size))
        average_reward = sum(item.reward for item in batch) / len(batch)
        return {
            "policy_loss": max(0.0, 1.0 - average_reward),
            "q_loss": max(0.0, 1.0 + average_reward),
            "average_reward": average_reward,
        }

    def save(self, destination: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        destination["policy_state"] = "demo"

    def load(self, source: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        _ = source

    @classmethod
    def from_factory(
        cls,
        factory: NetworkFactory,
        replay_buffer: BaseReplayBuffer,
        config: AgentConfig,
        **network_kwargs: Any,
    ) -> "DemoSACAgent":
        policy = factory.build_policy(**network_kwargs)
        q1, q2 = factory.build_q_functions(**network_kwargs)
        target_q1, target_q2 = factory.build_q_functions(**network_kwargs)
        return cls(policy, q1, q2, target_q1, target_q2, replay_buffer, config)


class DemoTrainer(Trainer):
    """Trainer that runs a short rollout using the demo agent and environment."""

    def run(self) -> None:
        state = self.environment.reset()
        for step in range(1, self.config.total_steps + 1):
            action = self.agent.act(state)
            transition = self.environment.step(action)
            self.agent.record(transition)

            metrics: MutableMapping[str, Any] = {
                "reward": transition.reward,
                "buffer_size": len(self.agent.replay_buffer),
            }

            if (
                step > self.config.warmup_steps
                and len(self.agent.replay_buffer) >= self.config.batch_size
            ):
                for _ in range(self.config.updates_per_step):
                    metrics.update(self.agent.update())

            self.log(metrics, step)
            print(
                f"Step {step:02d} | reward={metrics['reward']:.2f} "
                f"buffer={metrics['buffer_size']} "
                f"policy_loss={metrics.get('policy_loss', float('nan')):.2f}"
            )

            state = transition.next_state
            if transition.done:
                state = self.environment.reset()


def build_demo_components(article_path: Path, capacity: int) -> tuple[DemoSACAgent, DemoTrainer]:
    features = load_article_features(article_path)
    # Use the new delayed-reward long-read MDP environment by default
    environment = LongReadMDPEnvironment(features)
    replay_buffer = SimpleReplayBuffer(capacity)
    network_factory = DemoNetworkFactory(None, None, None)
    agent_config = AgentConfig()
    agent = DemoSACAgent.from_factory(network_factory, replay_buffer, agent_config)
    trainer_config = TrainerConfig(total_steps=12, warmup_steps=2, batch_size=4, updates_per_step=1)
    trainer = DemoTrainer(agent, environment, trainer_config)
    return agent, trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAC scaffolding demo")
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of interaction steps to simulate.",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=32,
        help="Maximum number of transitions stored in the replay buffer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    article_path = REPO_ROOT / "data" / "sample_article.txt"
    agent, trainer = build_demo_components(article_path, args.replay_capacity)
    trainer.config.total_steps = args.steps
    trainer.run()

    # Save agent snapshot
    snapshot: MutableMapping[str, Any] = {}
    agent.save(snapshot)
    OUT_DIR.mkdir(exist_ok=True)
    snapshot_path = OUT_DIR / "demo_agent_snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "agent_state": snapshot,
                "metadata": {"steps": args.steps, "replay_capacity": args.replay_capacity},
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved demo agent snapshot to {snapshot_path.relative_to(REPO_ROOT)}")

    # Export a distilled dataset of (s_t, a_t) pairs from the replay buffer
    dataset_path = OUT_DIR / "demo_expert_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as fh:
        for tr in getattr(agent, "replay_buffer")._storage:  # type: ignore[attr-defined]
            record = {
                "state": list(map(float, tr.state)),
                "action": list(map(float, tr.action)),
                "reward": float(tr.reward),
                "done": bool(tr.done),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved distilled dataset to {dataset_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
