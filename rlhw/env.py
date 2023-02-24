import gym
from typing import Callable, Dict, Any


def env_builder(
    env_name: str, params: Dict[str, Any] = {}
) -> Callable[[bool], gym.Env]:
    set_render = lambda render: "human" if render else None
    build_env = lambda render: gym.make(
        env_name, render_mode=set_render(render), **params
    )
    return build_env
