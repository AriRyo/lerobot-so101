# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ForceFeedbackConfig:
    enable: bool = False
    source: str = "current"
    gain: float = 0.015
    offset_limit: float = 5.0
    lowpass_tau_s: float = 0.05
    baseline_tau_s: float = 0.1
    deadband: float = 25.0
    max_step: float | None = None


class ForceFeedbackController:
    def __init__(self, joint_names: Iterable[str], config: ForceFeedbackConfig):
        self.config = config
        self.joint_names = list(joint_names)
        self._baseline: dict[str, float] = {joint: 0.0 for joint in self.joint_names}
        self._filtered: dict[str, float] = {joint: 0.0 for joint in self.joint_names}
        self._initialized = False

    def reset(self) -> None:
        self._baseline = {joint: 0.0 for joint in self.joint_names}
        self._filtered = {joint: 0.0 for joint in self.joint_names}
        self._initialized = False

    def update(self, signals: dict[str, float], dt_s: float) -> dict[str, float]:
        if not self.config.enable:
            return {joint: 0.0 for joint in self.joint_names}

        if not self._initialized:
            for joint in self.joint_names:
                self._baseline[joint] = float(signals.get(joint, 0.0))
                self._filtered[joint] = 0.0
            self._initialized = True

        outputs: dict[str, float] = {}
        for joint in self.joint_names:
            raw = float(signals.get(joint, 0.0))
            baseline = _lowpass(self._baseline[joint], raw, dt_s, self.config.baseline_tau_s)
            self._baseline[joint] = baseline

            excess = raw - baseline
            if abs(excess) < self.config.deadband:
                excess = 0.0

            target = excess * self.config.gain
            filtered = _lowpass(self._filtered[joint], target, dt_s, self.config.lowpass_tau_s)
            filtered = _clamp(filtered, -self.config.offset_limit, self.config.offset_limit)

            if self.config.max_step is not None:
                last = self._filtered[joint]
                filtered = _clamp(filtered, last - self.config.max_step, last + self.config.max_step)

            self._filtered[joint] = filtered
            outputs[joint] = filtered

        return outputs


def extract_force_signals(
    observation: dict[str, float], joint_names: Iterable[str], source: str
) -> dict[str, float]:
    suffix = f".{source}"
    signals: dict[str, float] = {}
    for joint in joint_names:
        key = f"{joint}{suffix}"
        if key in observation:
            signals[joint] = float(observation[key])
    return signals


def build_offset_feedback(offsets: dict[str, float]) -> dict[str, float]:
    return {f"{joint}.offset": value for joint, value in offsets.items()}


def _clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _lowpass(prev: float, target: float, dt_s: float, tau_s: float) -> float:
    if tau_s <= 0.0:
        return target
    if dt_s <= 0.0:
        return prev
    alpha = min(dt_s / tau_s, 1.0)
    return prev + alpha * (target - prev)
