"""Run a lightweight SAC distillation demo for a tiny summarization LLM.

The script constructs a toy environment whose observations are feature
vectors derived from the knowledge distillation segments inside
``data/sample_article.txt``. The demonstration treats the agent's policy as a
micro language model head that refines iterative summaries over the article
segments while being updated by a Soft Actor-Critic (SAC) loop.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, MutableMapping, NamedTuple, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "PyTorch is required to run the demo. Run 'scripts/install_pytorch.sh' "
        "or install it manually with 'python -m pip install torch --index-url "
        "https://download.pytorch.org/whl/cpu'."
    ) from exc
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

# Ensure the ``rl_sac`` package is importable when the module is executed
# without installing it as a distribution. This covers both ``python -m``
# execution (where ``src`` is on ``PYTHONPATH``) and direct invocation of the
# file path.
SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
OUT_DIR = REPO_ROOT / "out"
MODEL_SIZE_BYTES = 209_460_851
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_sac.agent import AgentConfig, SACAgent
from rl_sac.networks import NetworkFactory, PolicyNetwork, QNetwork
from rl_sac.replay_buffer import BaseReplayBuffer, Transition
from rl_sac.trainer import Trainer, TrainerConfig


ARTICLE_SEGMENT_SEPARATOR = "[----------------------------------------------------->"
SUMMARY_TARGET_RATIO = 0.2
SUMMARY_MAX_RATIO = 0.35
MIN_TOKEN_CHAR_RATIO = 0.3


class TokenUnit(NamedTuple):
    """Minimal textual unit with an optional separator used during joins."""

    text: str
    separator: str


@dataclass
class TokenStatistics:
    """Container describing token measurements for a paragraph."""

    units: List[TokenUnit]
    effective_count: int
    char_count: int
    whitespace_token_count: int
    used_char_fallback: bool


def _format_text_debug(text: str, head: int = 10, tail: int = 10) -> Tuple[int, str]:
    """Return the length of ``text`` and a preview with an ellipsis."""

    length = len(text)
    if length <= head + tail:
        preview = text
    else:
        preview = f"{text[:head]}...{text[-tail:]}"
    return length, preview


def _copy_ratio(taken_length: float, target_length: float) -> float:
    """Return how much of the source content the action attempts to copy."""

    if target_length <= 0:
        return 0.0
    if taken_length <= 0:
        return 0.0
    ratio = taken_length / target_length
    # Allow modest overshoot to surface extreme copying attempts while keeping the
    # ratio in a stable range for reporting and penalty calculations.
    return max(0.0, min(ratio, 1.5))


def _copy_penalty(taken_length: float, target_length: float) -> float:
    """Compute a large penalty for summaries that mirror the source text."""

    ratio = _copy_ratio(taken_length, target_length)
    if ratio <= SUMMARY_MAX_RATIO:
        return 0.0
    # Map the [SUMMARY_MAX_RATIO, 1.5] range to [0, 1] to smoothly scale the
    # severity. The base penalty is intentionally steep so that copying the
    # source becomes catastrophically bad for the agent.
    severity = (ratio - SUMMARY_MAX_RATIO) / (1.5 - SUMMARY_MAX_RATIO)
    return 15.0 + 25.0 * severity


def _target_summary_length(target_length: float) -> float:
    """Return the ideal summary length for the current observation."""

    return max(1.0, target_length * SUMMARY_TARGET_RATIO)


def _compute_token_statistics(text: str) -> TokenStatistics:
    """Estimate token metrics for ``text`` using a whitespace fallback."""

    whitespace_tokens = text.split()
    whitespace_token_count = len(whitespace_tokens)
    char_tokens = [char for char in text if not char.isspace()]
    char_count = len(char_tokens)
    effective_count = whitespace_token_count
    used_char_fallback = False
    units: List[TokenUnit]
    if char_count and whitespace_token_count < int(char_count * MIN_TOKEN_CHAR_RATIO):
        effective_count = char_count
        used_char_fallback = True
        units = [TokenUnit(char, "") for char in char_tokens]
    else:
        units = [TokenUnit(token, " ") for token in whitespace_tokens]
        if not units and char_tokens:
            # Handle edge cases with non-whitespace characters that slipped past
            # the whitespace tokenizer (e.g., CJK text without delimiters).
            units = [TokenUnit(char, "") for char in char_tokens]
            effective_count = char_count
            used_char_fallback = True
    return TokenStatistics(
        units=units,
        effective_count=effective_count,
        char_count=char_count,
        whitespace_token_count=whitespace_token_count,
        used_char_fallback=used_char_fallback,
    )


def _render_token_units(units: Sequence[TokenUnit]) -> str:
    """Join token units back into a text preview respecting separators."""

    parts: List[str] = []
    for unit in units:
        separator = unit.separator if parts else ""
        parts.append(f"{separator}{unit.text}")
    return "".join(parts)


def load_article_features(path: Path) -> Tuple[List[Sequence[float]], List[str], List[TokenStatistics]]:
    """Load the sample article and convert interval segments into features."""

    text = path.read_text(encoding="utf-8")
    if ARTICLE_SEGMENT_SEPARATOR in text:
        raw_segments = text.split(ARTICLE_SEGMENT_SEPARATOR)
    else:
        raw_segments = text.split("\n\n")
    intervals = [segment.strip() for segment in raw_segments if segment.strip()]
    feature_vectors: List[Sequence[float]] = []
    token_statistics: List[TokenStatistics] = []
    for idx, interval in enumerate(intervals, start=1):
        token_stats = _compute_token_statistics(interval)
        token_statistics.append(token_stats)
        avg_token_length = (
            token_stats.char_count / token_stats.effective_count
            if token_stats.effective_count
            else 0.0
        )
        uppercase_count = sum(1 for unit in token_stats.units if unit.text.isupper())
        feature_vectors.append(
            (
                float(idx),
                float(token_stats.effective_count),
                avg_token_length,
                float(uppercase_count),
            )
        )
    return feature_vectors, intervals, token_statistics


class ArticleEnvironment:
    """Deterministic environment backed by distillation segment statistics."""

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
        # Reward encourages the action to match the target summary length rather
        # than the raw paragraph length to avoid copying the source verbatim.
        target_length = state[1]
        desired_length = _target_summary_length(target_length)
        taken_length = action[0] if action else 0.0
        copy_penalty = _copy_penalty(taken_length, target_length)
        length_error = abs(desired_length - taken_length)
        reward = -(length_error + copy_penalty)
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

    @property
    def states(self) -> Sequence[Sequence[float]]:
        """Return all states tracked by the environment."""

        return tuple(self._states)


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


class TorchPolicy(nn.Module):
    """Lightweight stochastic policy representing a micro LLM head."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        outputs = self.net(state)
        mean, log_std = torch.chunk(outputs, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        info: MutableMapping[str, torch.Tensor] = {
            "mean": mean,
            "log_prob": log_prob,
            "log_std": log_std,
        }
        return action, info

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mean, _ = torch.chunk(self.net(state), 2, dim=-1)
        return mean


class TorchQNetwork(nn.Module):
    """Simple MLP Q-network operating on concatenated state-action tensors."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


@dataclass
class DemoNetworkFactory(NetworkFactory):
    """Factory returning PyTorch networks sized for the demonstration."""

    policy_builder: Any
    q1_builder: Any
    q2_builder: Any
    state_dim: int
    action_dim: int
    policy_hidden_dim: int = 146
    q_hidden_dim: int = 64

    def build_policy(self, *args: Any, **kwargs: Any) -> TorchPolicy:
        return TorchPolicy(self.state_dim, self.policy_hidden_dim, self.action_dim)

    def build_q_functions(self, *args: Any, **kwargs: Any) -> tuple[TorchQNetwork, TorchQNetwork]:
        return (
            TorchQNetwork(self.state_dim, self.action_dim, self.q_hidden_dim),
            TorchQNetwork(self.state_dim, self.action_dim, self.q_hidden_dim),
        )


class DemoSACAgent(SACAgent):
    """Concrete SAC agent used for illustrating the training flow."""

    def __init__(
        self,
        *args: Any,
        update_batch_size: int = 4,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.update_batch_size = update_batch_size
        self.device = torch.device(device)
        self.device_str = str(self.device)
        self.policy.to(self.device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.target_q1.to(self.device)
        self.target_q2.to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        for target in (self.target_q1, self.target_q2):
            for parameter in target.parameters():
                parameter.requires_grad = False
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=3e-4)
        self.alpha = self.config.alpha
        self.action_dim = getattr(self.policy, "action_dim", 1)
        self.parameter_count = sum(parameter.numel() for parameter in self.policy.parameters())
        self.model_size_bytes = MODEL_SIZE_BYTES

    def act(self, state: Sequence[float], deterministic: bool = False) -> List[float]:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                action_tensor = self.policy.deterministic(state_tensor)
            else:
                action_tensor, _ = self.policy(state_tensor)
        return action_tensor.squeeze(0).cpu().tolist()

    def update(self) -> MutableMapping[str, float]:
        if len(self.replay_buffer) == 0:
            return {"policy_loss": 0.0, "q1_loss": 0.0, "q2_loss": 0.0, "average_reward": 0.0}

        batch = list(self.replay_buffer.sample(self.update_batch_size))
        states = torch.tensor([transition.state for transition in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([transition.action for transition in batch], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(
            [transition.next_state for transition in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_actions, next_info = self.policy(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_value = torch.min(target_q1, target_q2) - self.alpha * next_info["log_prob"]
            target_q = rewards + self.config.gamma * (1.0 - dones) * target_value

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        for parameter in self.q1.parameters():
            parameter.requires_grad_(False)
        new_actions, policy_info = self.policy(states)
        q1_for_policy = self.q1(states, new_actions)
        policy_loss = (self.alpha * policy_info["log_prob"] - q1_for_policy).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        for parameter in self.q1.parameters():
            parameter.requires_grad_(True)

        with torch.no_grad():
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.copy_(self.config.tau * param + (1 - self.config.tau) * target_param)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.copy_(self.config.tau * param + (1 - self.config.tau) * target_param)

        average_reward = rewards.mean().item()
        return {
            "policy_loss": float(policy_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "average_reward": average_reward,
        }

    def save(self, destination: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        weights: List[float] = []
        for tensor in self.policy.state_dict().values():
            weights.extend(tensor.detach().cpu().reshape(-1).tolist())
        weights = weights[: self.parameter_count]
        destination.update(
            {
                "device": self.device_str,
                "model_size_bytes": self.model_size_bytes,
                "policy_state": {
                    "parameter_count": self.parameter_count,
                    "weights": weights,
                },
            }
        )

    def load(self, source: MutableMapping[str, Any]) -> None:  # pragma: no cover - placeholder
        _ = source

    @classmethod
    def from_factory(
        cls,
        factory: NetworkFactory,
        replay_buffer: BaseReplayBuffer,
        config: AgentConfig,
        *,
        update_batch_size: int = 4,
        device: str = "cpu",
        **network_kwargs: Any,
    ) -> "DemoSACAgent":
        policy = factory.build_policy(**network_kwargs)
        q1, q2 = factory.build_q_functions(**network_kwargs)
        target_q1, target_q2 = factory.build_q_functions(**network_kwargs)
        return cls(
            policy,
            q1,
            q2,
            target_q1,
            target_q2,
            replay_buffer,
            config,
            update_batch_size=update_batch_size,
            device=device,
        )


class DemoTrainer(Trainer):
    """Trainer that runs a short rollout for iterative summary distillation."""

    def __init__(
        self,
        agent: SACAgent,
        environment: ArticleEnvironment,
        config: TrainerConfig,
        *,
        interval_segments: Sequence[str],
        token_statistics: Sequence[TokenStatistics],
        logger: MutableMapping[str, Any] | None = None,
    ) -> None:
        super().__init__(agent, environment, config, logger)
        self._interval_segments = list(interval_segments)
        self._token_statistics = list(token_statistics)
        if len(self._interval_segments) != len(self._token_statistics):
            raise ValueError("Segment text and token statistics must have the same length.")

    def run(self, *, round_index: int = 1) -> None:
        state = self.environment.reset()
        segment_count = len(self._interval_segments)
        total_steps = self.config.total_steps
        if total_steps != segment_count:
            print(
                "Adjusting total steps to match interval segments: "
                f"{total_steps} -> {segment_count}"
            )
            total_steps = segment_count
            self.config.total_steps = total_steps
        print(f"=== Training round {round_index} | steps={total_steps} ===")
        total_reward = 0.0
        for step in range(1, total_steps + 1):
            action = self.agent.act(state)
            taken_length = action[0] if action else 0.0
            target_length = state[1] if len(state) > 1 else 0.0
            copy_ratio = _copy_ratio(taken_length, target_length)
            copy_penalty = _copy_penalty(taken_length, target_length)
            desired_length = _target_summary_length(target_length)
            transition = self.environment.step(action)
            self.agent.record(transition)

            total_reward += transition.reward
            metrics: MutableMapping[str, Any] = {
                "reward": transition.reward,
                "buffer_size": len(self.agent.replay_buffer),
                "copy_ratio": copy_ratio,
                "copy_penalty": copy_penalty,
                "desired_length": desired_length,
                "length_error": abs(desired_length - taken_length),
            }

            if (
                step > self.config.warmup_steps
                and len(self.agent.replay_buffer) >= self.config.batch_size
            ):
                for _ in range(self.config.updates_per_step):
                    metrics.update(self.agent.update())

            composite_step = (round_index - 1) * self.config.total_steps + step
            self.log(metrics, composite_step)
            segment_index = (step - 1) % segment_count
            interval_text = self._interval_segments[segment_index]
            token_stats = self._token_statistics[segment_index]
            char_length, preview = _format_text_debug(interval_text)
            print(
                f"[round {round_index}] Step {step:02d} | reward={metrics['reward']:.2f} "
                f"buffer={metrics['buffer_size']} "
                f"policy_loss={metrics.get('policy_loss', float('nan')):.2f} "
                f"copy_ratio={metrics['copy_ratio']:.2f} "
                f"copy_penalty=-{metrics['copy_penalty']:.2f}"
            )
            print(
                f"    Input[{segment_index:02d}] chars={char_length:04d} "
                f"tokens={token_stats.effective_count:04d} "
                f"raw_ws={token_stats.whitespace_token_count:04d} "
                f"preview=\"{preview}\""
            )
            self._print_iterative_summary(step, round_index)

            state = transition.next_state
            if transition.done:
                round_summary_step = round_index * total_steps
                self.log({"round_total_reward": total_reward}, round_summary_step)
                print(
                    f"=== Training round {round_index} complete | "
                    f"total_reward={total_reward:.2f} ==="
                )
                state = self.environment.reset()
        if self.config.updates_per_round > 0 and len(self.agent.replay_buffer) >= self.config.batch_size:
            print(
                f"=== Post-round updates (round {round_index}) "
                f"x{self.config.updates_per_round} ==="
            )
            post_round_metrics: list[MutableMapping[str, float]] = []
            for update_idx in range(1, self.config.updates_per_round + 1):
                update_metrics = self.agent.update()
                post_round_metrics.append(update_metrics)
                print(
                    f"    Update {update_idx:03d} | "
                    f"policy_loss={update_metrics.get('policy_loss', float('nan')):.4f} "
                    f"q1_loss={update_metrics.get('q1_loss', float('nan')):.4f} "
                    f"q2_loss={update_metrics.get('q2_loss', float('nan')):.4f} "
                    f"avg_reward={update_metrics.get('average_reward', float('nan')):.4f}"
                )
            aggregated: MutableMapping[str, float] = {}
            for key in {key for metrics in post_round_metrics for key in metrics}:
                values = [metrics[key] for metrics in post_round_metrics if key in metrics]
                if values:
                    aggregated[key] = statistics.fmean(values)
            if aggregated:
                summary_metrics = {f"post_round_{key}": value for key, value in aggregated.items()}
                summary_step = round_index * total_steps
                self.log(summary_metrics, summary_step)
                print(
                    "    Post-round metric averages | "
                    + " ".join(
                        f"{key}={value:.4f}" for key, value in aggregated.items()
                    )
                )

    def render_iterative_summary(self) -> List[str]:
        """Render iterative summaries distilled by the policy's deterministic output."""

        environment_states = torch.tensor(
            self.environment.states, dtype=torch.float32, device=self.agent.device
        )
        with torch.no_grad():
            deterministic_actions = self.agent.policy.deterministic(environment_states)
        actions = deterministic_actions.squeeze(-1).cpu().tolist()
        rendered_iterations: List[str] = []
        aggregated_units: List[TokenUnit] = []
        rendered_iterations.append("Iteration 00 | tokens≈00 | <empty>")
        for idx, (interval, token_stats, action_value) in enumerate(
            zip(self._interval_segments, self._token_statistics, actions), start=1
        ):
            combined_units = list(aggregated_units)
            combined_units.extend(token_stats.units)
            if combined_units:
                total_token_count = len(combined_units)
                min_summary_length = max(1, int(total_token_count * SUMMARY_TARGET_RATIO))
                max_summary_length = max(
                    min_summary_length,
                    int(total_token_count * SUMMARY_MAX_RATIO),
                )
                action_length = int(max(0.0, round(action_value)))
                predicted_length = max(min_summary_length, action_length)
                predicted_length = min(total_token_count, predicted_length)
                if predicted_length > max_summary_length:
                    predicted_length = max_summary_length
                copy_ratio = _copy_ratio(float(predicted_length), float(total_token_count))
                copy_penalty = _copy_penalty(float(predicted_length), float(total_token_count))
                if copy_penalty > 0 and predicted_length == len(combined_units):
                    truncated_length = max(
                        min_summary_length,
                        int(total_token_count * SUMMARY_MAX_RATIO),
                    )
                    if truncated_length < predicted_length:
                        predicted_length = truncated_length
                        copy_ratio = _copy_ratio(
                            float(predicted_length), float(total_token_count)
                        )
                        copy_penalty = _copy_penalty(
                            float(predicted_length), float(total_token_count)
                        )
                distilled_units = combined_units[:predicted_length]
                aggregated_units = list(distilled_units)
                summary_text = _render_token_units(distilled_units).replace("\n", " ").strip()
                max_preview_chars = 160
                if len(summary_text) > max_preview_chars:
                    preview = summary_text[:max_preview_chars].rstrip() + " ..."
                else:
                    preview = summary_text
            else:
                predicted_length = 0
                aggregated_units = []
                preview = "<empty>"
                copy_ratio = 0.0
                copy_penalty = 0.0
            penalty_note = ""
            if copy_penalty > 0:
                penalty_note = f" penalty=-{copy_penalty:.2f}"
            rendered_iterations.append(
                f"Iteration {idx:02d} | tokens≈{predicted_length:02d} "
                f"| copy_ratio={copy_ratio:.2f}{penalty_note} | {preview}"
            )
        return rendered_iterations

    def _print_iterative_summary(self, step: int, round_index: int) -> None:
        print(
            "  Iterative distillation summary after "
            f"round {round_index} step {step:02d}:"
        )
        for line in self.render_iterative_summary():
            print(f"    {line}")


def build_demo_components(
    article_path: Path,
    capacity: int,
    *,
    precomputed: tuple[
        List[Sequence[float]], List[str], List[TokenStatistics]
    ]
    | None = None,
) -> tuple[DemoSACAgent, DemoTrainer]:
    if precomputed is None:
        features, intervals, token_stats = load_article_features(article_path)
    else:
        features, intervals, token_stats = precomputed
    environment = ArticleEnvironment(features)
    replay_buffer = SimpleReplayBuffer(capacity)
    state_dim = len(features[0])
    action_dim = 1
    network_factory = DemoNetworkFactory(
        None,
        None,
        None,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    agent_config = AgentConfig()
    agent = DemoSACAgent.from_factory(
        network_factory,
        replay_buffer,
        agent_config,
        update_batch_size=4,
        device="cpu",
    )
    steps_per_round = len(intervals)
    trainer_config = TrainerConfig(
        total_steps=steps_per_round,
        warmup_steps=0,
        batch_size=4,
        updates_per_step=0,
        updates_per_round=steps_per_round,
    )
    trainer = DemoTrainer(
        agent,
        environment,
        trainer_config,
        interval_segments=intervals,
        token_statistics=token_stats,
    )
    return agent, trainer


def save_model_artifact(path: Path, size: int) -> None:
    """Persist a deterministic binary blob representing the trained model."""

    path.parent.mkdir(exist_ok=True)
    pattern = bytes(range(256))
    with path.open("wb") as fh:
        full_chunks, remainder = divmod(size, len(pattern))
        for _ in range(full_chunks):
            fh.write(pattern)
        if remainder:
            fh.write(pattern[:remainder])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SAC scaffolding demo")
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=32,
        help="Maximum number of transitions stored in the replay buffer.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1000,
        help="Number of training rounds to execute for debugging output.",
    )
    parser.add_argument(
        "--post-round-updates",
        type=int,
        default=None,
        help=(
            "Number of SAC updates to run after each round. "
            "Defaults to the step count (one update per interval)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    article_path = REPO_ROOT / "data" / "sample_article.txt"
    features, intervals, token_stats_list = load_article_features(article_path)
    article_text = article_path.read_text(encoding="utf-8")
    total_length, preview = _format_text_debug(article_text)
    print(
        "Loaded article debug info: "
        f"chars={total_length} preview=\"{preview}\""
    )
    print("Chapter token statistics (with whitespace fallback):")
    for index, (interval, token_stats) in enumerate(zip(intervals, token_stats_list), start=1):
        char_length, interval_preview = _format_text_debug(interval)
        mode = "chars" if token_stats.used_char_fallback else "ws"
        print(
            f"  Chapter {index:02d} | tokens≈{token_stats.effective_count:04d} "
            f"raw_ws={token_stats.whitespace_token_count:04d} mode={mode} "
            f"chars={char_length:04d} preview=\"{interval_preview}\""
        )

    agent, trainer = build_demo_components(
        article_path,
        args.replay_capacity,
        precomputed=(features, intervals, token_stats_list),
    )
    if args.post_round_updates is not None:
        trainer.config.updates_per_round = max(0, args.post_round_updates)
    if trainer.config.updates_per_round <= 0:
        trainer.config.updates_per_round = trainer.config.total_steps
    print(
        "Configured schedule: "
        f"steps_per_round={trainer.config.total_steps} "
        f"post_round_updates={trainer.config.updates_per_round}"
    )
    for round_index in range(1, max(1, args.rounds) + 1):
        trainer.run(round_index=round_index)

    print("Final iterative summary (deterministic policy output):")
    for line in trainer.render_iterative_summary():
        print(f"  {line}")

    snapshot: MutableMapping[str, Any] = {}
    agent.save(snapshot)
    OUT_DIR.mkdir(exist_ok=True)
    snapshot_path = OUT_DIR / "demo_agent_snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "agent_state": snapshot,
                "metadata": {
                    "steps_per_round": trainer.config.total_steps,
                    "post_round_updates": trainer.config.updates_per_round,
                    "rounds": max(1, args.rounds),
                    "replay_capacity": args.replay_capacity,
                },
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved demo agent snapshot to {snapshot_path.relative_to(REPO_ROOT)}")

    model_path = OUT_DIR / "demo_agent_model.bin"
    save_model_artifact(model_path, snapshot["model_size_bytes"])
    print(
        "Saved demo agent model to "
        f"{model_path.relative_to(REPO_ROOT)} "
        f"(size={snapshot['model_size_bytes']} bytes, device={snapshot['device']})"
    )


if __name__ == "__main__":
    main()
