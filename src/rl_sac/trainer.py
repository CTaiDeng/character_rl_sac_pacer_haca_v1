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

"""Training loop utilities for SAC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Protocol

from .agent import SACAgent
from .replay_buffer import Transition


class Environment(Protocol):
    """Minimal environment protocol compatible with the trainer."""

    def reset(self) -> Any:
        """Reset the environment and return the initial state."""

    def step(self, action: Any) -> Transition:
        """Advance the environment using ``action`` and return a transition."""


@dataclass
class TrainerConfig:
    """High-level configuration controlling the training procedure."""

    total_steps: int = 1_000_000
    warmup_steps: int = 1_000
    batch_size: int = 256
    updates_per_step: int = 1
    updates_per_round: int = 0


class Trainer:
    """Skeleton implementation of an offline training loop."""

    def __init__(
        self,
        agent: SACAgent,
        environment: Environment,
        config: TrainerConfig,
        logger: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.config = config
        self.logger = logger if logger is not None else {}

    def run(self) -> None:
        """Execute the training loop.

        This placeholder outlines the warmup, interaction, and update phases of
        a typical SAC training loop. Implementations should fill in the missing
        logic for collecting transitions, updating the agent, and logging
        metrics.
        """

        raise NotImplementedError("Implement trainer execution loop.")

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        """Record ``metrics`` at a given ``step`` in the logger."""

        self.logger.setdefault(step, {}).update(metrics)


__all__ = ["Environment", "TrainerConfig", "Trainer"]
