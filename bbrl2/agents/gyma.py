# coding=utf-8
#
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import torch
from gymnasium import Env, Space
from gymnasium.core import ActType, ObsType
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import AutoResetWrapper
from torch import nn, Tensor

from bbrl2 import SeedableAgent, SerializableAgent, TimeAgent


def _convert_action(action: Tensor) -> Union[int, np.ndarray]:
    if len(action.size()) == 0:
        action = action.item()
        assert isinstance(action, int)
    else:
        action = np.array(action.tolist())
    return action


def _format_frame(
    frame: Union[Dict[str, Tensor], List[Tensor], np.ndarray, Tensor, bool, int, float]
) -> Union[Tensor, Dict[str, Tensor]]:
    if isinstance(frame, Dict):
        r = {}
        for k in frame:
            r[k] = _format_frame(frame[k])
        return r
    elif isinstance(frame, List):
        t = torch.tensor(frame).unsqueeze(0)
        if t.dtype == torch.float64 or t.dtype == torch.float32:
            t = t.float()
        else:
            t = t.long()
        return t
    elif isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame).unsqueeze(0)
        if t.dtype == torch.float64 or t.dtype == torch.float32:
            t = t.float()
        else:
            t = t.long()
        return t
    elif isinstance(frame, Tensor):
        return frame.unsqueeze(0)
    elif isinstance(frame, bool):
        return torch.tensor([frame]).bool()
    elif isinstance(frame, int):
        return torch.tensor([frame]).long()
    elif isinstance(frame, float):
        return torch.tensor([frame]).float()

    else:
        try:
            # Check if it is a LazyFrame from OpenAI Baselines
            o = torch.from_numpy(frame.__array__()).unsqueeze(0).float()
            return o
        except TypeError:
            assert False


def _torch_type(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: d[k].float() if torch.is_floating_point(d[k]) else d[k] for k in d}


def _torch_cat_dict(d: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r


class GymAgent(TimeAgent, SeedableAgent, SerializableAgent, ABC):
    default_seed = 42

    def __init__(
        self,
        input_string: str = "action",
        output_string: str = "env/",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.ghost_params: nn.Parameter = nn.Parameter(torch.randn(()))

        self.input: str = input_string
        self.output: str = output_string
        self._timestep_from_reset: int = 0
        self._nb_reset: int = 1

        self.observation_space: Optional[Space[ObsType]] = None
        self.action_space: Optional[Space[ActType]] = None

    def forward(self, t: int, *args, **kwargs) -> None:
        if self._seed is None:
            self.seed(self.default_seed)
        if t == 0:
            self._timestep_from_reset = 1
            self._nb_reset += 1
        else:
            self._timestep_from_reset += 1

    def set_obs(self, observations: Dict[str, Tensor], t: int) -> None:
        for k in observations:
            self.set(
                (self.output + k, t),
                observations[k].to(self.ghost_params.device),
            )

    def get_observation_space(self) -> Space[ObsType]:
        """Return the observation space of the environment"""
        if self.observation_space is None:
            raise ValueError("The observation space is not defined")
        return self.observation_space

    def get_action_space(self) -> Space[ActType]:
        """Return the action space of the environment"""
        if self.action_space is None:
            raise ValueError("The action space is not defined")
        return self.action_space


class ParallelGymAgent(GymAgent):
    """Create an Agent from a gymnasium environment
    The default 
    To create an auto-reset ParallelGymAgent, use the gymnasium `AutoResetWrapper` in the make_env_fn
    """

    def __init__(
        self,
        make_env_fn: Callable[[Optional[Dict[str, Any]]], Env],
        num_envs: int,
        make_env_args: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """Create an agent from a Gymnasium environment

        Args:
            make_env_fn ([function that returns a gymnasium.Env]): The function to create a single gymnasium environment
            num_envs ([int]): The number of environments to create.
            make_env_args (dict): The arguments of the function that creates a gymnasium.Env
            input_string (str, optional): [the name of the action variable in the workspace]. Defaults to "action".
            output_string (str, optional): [the output prefix of the environment]. Defaults to "env/".
            max_episode_steps (int, optional): Max number of steps per episode. Defaults to None (never ends)
        """
        super().__init__(*args, **kwargs)
        assert num_envs > 0, "n_envs must be > 0"

        self.make_env_fn: Callable[[Optional[Dict[str, Any]]], Env] = make_env_fn
        self.num_envs: int = num_envs

        self.envs: List[Env] = []
        self.cumulated_reward: Dict[int, float] = {}

        self._timestep: Tensor
        self._is_autoreset: bool = False
        self._last_frame: Dict[int] = {}

        args: Dict[str, Any] = make_env_args if make_env_args is not None else {}
        self._initialize_envs(num_envs=num_envs, make_env_args=args)

    def _initialize_envs(self, num_envs, make_env_args: Dict[str, Any]):
        self.envs = [self.make_env_fn(**make_env_args) for _ in range(num_envs)]
        self._timestep = torch.zeros(len(self.envs), dtype=torch.long)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        unwrapped_env = self.envs[0].unwrapped
        wrapper = self.envs[0]
        while type(wrapper) is not type(unwrapped_env):
            if type(wrapper) == AutoResetWrapper:
                self._is_autoreset = True
            wrapper = wrapper.env

        if self._is_autoreset and self._max_episode_steps is None:
            raise ValueError(
                "AutoResetWrapper without max_episode_steps argument given will (probably) never"
                "stop the GymAgent if wrapped with a TemporalAgent"
            )

    def _reset(self, k: int) -> Dict[str, Tensor]:
        env: Env = self.envs[k]
        self.cumulated_reward[k] = 0.0

        s: int = self._timestep_from_reset * self.num_envs * self._nb_reset * self._seed

        s += (k + 1) * (self._timestep[k].item() + 1 if self._is_autoreset else 1)

        o, info = env.reset(seed=s)
        observation: Union[Tensor, Dict[str, Tensor]] = _format_frame(o)

        self._timestep[k] = 0

        if isinstance(observation, Tensor):
            observation = {"env_obs": observation}
        elif isinstance(observation, dict):
            pass
        else:
            raise ValueError(
                f"Observation must be a torch.Tensor or a dict, not {type(observation)}"
            )

        ret: Dict[str, Tensor] = {
            **observation,
            "terminated": torch.tensor([False]),
            "truncated": torch.tensor([False]),
            "reward": torch.tensor([0.0]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self._timestep[k]]),
        }
        self._last_frame[k] = ret
        return _torch_type(ret)

    def _step(self, k: int, action: Tensor):
        env = self.envs[k]
        action: Union[int, np.ndarray[int]] = _convert_action(action)

        obs, reward, terminated, truncated, info = env.step(action)

        self.cumulated_reward[k] += reward
        observation: Union[Tensor, Dict[str, Tensor]] = _format_frame(obs)

        if isinstance(observation, Tensor):
            observation = {"env_obs": observation}
        elif isinstance(observation, dict):
            pass
        else:
            raise ValueError(
                f"Observation must be a torch.Tensor or a dict, not {type(observation)}"
            )

        self._timestep[k] += 1

        ret: Dict[str, Tensor] = {
            **observation,
            "terminated": torch.tensor([terminated]),
            "truncated": torch.tensor([truncated]),
            "reward": torch.tensor([reward]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self._timestep[k]]),
        }
        self._last_frame[k] = ret
        return _torch_type(ret)

    def forward(self, t: int = 0, **kwargs) -> None:
        """Do one step by reading the `action` at t-1
        If t==0, environments are reset
        If render is True, then the output of env.render() is written as env/rendering
        """
        super().forward(t, **kwargs)

        observations = []
        if t == 0:
            for k, env in enumerate(self.envs):
                observations.append(self._reset(k))
        else:
            action = self.get((self.input, t - 1))
            assert action.size()[0] == self.num_envs, "Incompatible number of envs"

            for k, env in enumerate(self.envs):
                if self._is_autoreset or not self._last_frame[k]["terminated"]:
                    observations.append(self._step(k, action[k]))
                else:
                    observations.append(self._last_frame[k])
        self.set_obs(observations=_torch_cat_dict(observations), t=t)


class VecGymAgent(GymAgent):
    def __init__(
        self,
        make_envs_fn: Callable[[Optional[Dict[str, Any]]], VectorEnv],
        vec_env_args: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        args: Dict[str, Any] = vec_env_args or {}
        self.envs: VectorEnv = make_envs_fn(**args)

        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

        self.cumulated_reward: Tensor = torch.zeros(self.envs.num_envs)

    def forward(self, t: int, **kwargs) -> None:
        super().forward(t, **kwargs)

        if t == 0:
            s: int = self._seed * self._nb_reset
            obs, infos = self.envs.reset(seed=s)
            termination = torch.tensor([False] * self.envs.num_envs)
            truncation = torch.tensor([False] * self.envs.num_envs)
            rewards = torch.tensor([0.0] * self.envs.num_envs)
            self.cumulated_reward = torch.zeros(self.envs.num_envs)
        else:
            action = self.get((self.input, t - 1))
            assert (
                action.size()[0] == self.envs.num_envs
            ), "Incompatible number of actions"
            converted_action: Union[int, np.ndarray[int]] = _convert_action(action)
            obs, rewards, termination, truncation, infos = self.envs.step(
                converted_action
            )
            rewards = torch.tensor(rewards).float()
            termination = torch.tensor(termination)
            truncation = torch.tensor(truncation)
            self.cumulated_reward = self.cumulated_reward + rewards

        observation: Union[Tensor, Dict[str, Tensor]] = _format_frame(obs)

        if not isinstance(observation, Tensor):
            raise ValueError("Observation can't be an OrderedDict in a VecEnv")

        ret: Dict[str, Tensor] = {
            "env_obs": observation.squeeze(0),
            "terminated": termination,
            "truncated": truncation,
            "reward": rewards,
            "cumulated_reward": self.cumulated_reward,
        }
        self.set_obs(observations=ret, t=t)


class ImageGymAgent(GymAgent, ABC):
    """
    GymAgent compatible with image observations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def serialize(self):
        """Return a serializable GymAgent without the environments"""
        copied_agent = copy.copy(self)
        copied_agent.envs = None
        return copied_agent


class VecImageGymAgent(VecGymAgent, ImageGymAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ParallelImageGymAgent(ParallelGymAgent, ImageGymAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class NoAutoResetGymAgent(ParallelGymAgent):
    """
    Standard GymAgent for NoAutoReset environments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AutoResetGymAgent(ParallelGymAgent):
    """
    Standard GymAgent for AutoReset environments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

