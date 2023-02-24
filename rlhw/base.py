from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple

import gym
import numpy.typing as npt


EnvStepType = Tuple[npt.NDArray, int, float, npt.NDArray, bool]


class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, build_env: Callable[[bool], gym.Env], params: Dict[str, Any]):
        self.build_env = build_env
        self.params = params

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def run(self):
        pass
