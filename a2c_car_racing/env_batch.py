"""
Батч сред для A2C: последовательный EnvBatch и параллельный ParallelEnvBatch.
По мотивам https://github.com/yandexdataschool/Practical_RL/blob/master/week06_policy_based/env_batch.py
"""

from multiprocessing import Pipe, Process

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Space


class SpaceBatch(Space):
    """Пространство для батча одинаковых пространств (obs или action)."""

    def __init__(self, spaces):
        first_type = type(spaces[0])
        first_shape = spaces[0].shape
        first_dtype = spaces[0].dtype
        for space in spaces:
            if not isinstance(space, first_type):
                raise TypeError(
                    "spaces have different types: {}, {}".format(
                        first_type, type(space)
                    )
                )
            if first_shape != space.shape:
                raise ValueError(
                    "spaces have different shapes: {}, {}".format(
                        first_shape, space.shape
                    )
                )
            if first_dtype != space.dtype:
                raise ValueError(
                    "spaces have different data types: {}, {}".format(
                        first_dtype, space.dtype
                    )
                )
        self.spaces = spaces
        super().__init__(shape=first_shape, dtype=first_dtype)

    def sample(self):
        return np.stack([space.sample() for space in self.spaces])

    def __getattr__(self, attr):
        return getattr(self.spaces[0], attr)


class EnvBatch(Env):
    """Последовательный батч сред: step/reset вызываются по очереди для каждого env."""

    def __init__(self, make_env, nenvs=None):
        make_env_functions = self._get_make_env_functions(make_env, nenvs)
        self._envs = [make_env() for make_env in make_env_functions]
        self._nenvs = len(self._envs)
        self.observation_space = SpaceBatch(
            [env.observation_space for env in self._envs]
        )
        self.action_space = SpaceBatch([env.action_space for env in self._envs])

    def _get_make_env_functions(self, make_env, nenvs):
        if nenvs is None and not isinstance(make_env, list):
            raise ValueError("When nenvs is None make_env must be a list of callables")
        if nenvs is not None and not callable(make_env):
            raise ValueError("When nenvs is not None make_env must be callable")
        if nenvs is not None:
            make_env = [make_env for _ in range(nenvs)]
        return make_env

    @property
    def nenvs(self):
        return self._nenvs

    @property
    def envs(self):
        return self._envs

    def _check_actions(self, actions):
        if len(actions) != self._nenvs:
            raise ValueError(
                "number of actions is not equal to number of envs: "
                "len(actions)={}, nenvs={}".format(len(actions), self._nenvs)
            )

    def step(self, actions):
        self._check_actions(actions)
        observations, rewards, terminated_list, truncated_list, infos = (
            [],
            [],
            [],
            [],
            [],
        )
        for env, action in zip(self._envs, actions):
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
            observations.append(obs)
            rewards.append(rew)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            infos.append(info)
        return (
            np.stack(observations),
            np.stack(rewards),
            np.stack(terminated_list),
            np.stack(truncated_list),
            infos,
        )

    def reset(self, **kwargs):
        observations, infos = [], []
        for env in self._envs:
            obs, info = env.reset(**kwargs)
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def close(self):
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


def _worker(parent_connection, worker_connection, make_env, send_spaces=True):
    """Воркер для ParallelEnvBatch: в подпроцессе держит одну среду."""
    parent_connection.close()
    env = make_env()
    if send_spaces:
        worker_connection.send((env.observation_space, env.action_space))
    while True:
        cmd, data = worker_connection.recv()
        if cmd == "step":
            obs, rew, terminated, truncated, info = env.step(data)
            if terminated or truncated:
                obs, info = env.reset()
            worker_connection.send((obs, rew, terminated, truncated, info))
        elif cmd == "reset":
            obs, info = env.reset(seed=data)
            worker_connection.send((obs, info))
        elif cmd == "close":
            env.close()
            worker_connection.close()
            break
        else:
            raise NotImplementedError("Unknown command %s" % cmd)


class ParallelEnvBatch(Env):
    """
    Параллельный батч сред (отдельный процесс на каждую среду).
    Ускоряет сбор данных за счёт параллельного step().
    """

    def __init__(self, make_env, nenvs, seeds=None):
        if not callable(make_env):
            raise ValueError("make_env must be callable")
        self._nenvs = nenvs
        make_env_functions = [make_env for _ in range(nenvs)]
        self._seeds = seeds if seeds is not None else list(range(nenvs))
        if len(self._seeds) != nenvs:
            self._seeds = (self._seeds * (nenvs // len(self._seeds) + 1))[:nenvs]

        self._parent_connections, self._worker_connections = zip(
            *[Pipe() for _ in range(nenvs)]
        )
        self._processes = [
            Process(
                target=_worker,
                args=(parent_conn, worker_conn, make_env_functions[i]),
                daemon=True,
            )
            for i, (parent_conn, worker_conn) in enumerate(
                zip(self._parent_connections, self._worker_connections)
            )
        ]
        for p in self._processes:
            p.start()
        self._closed = False

        for conn in self._worker_connections:
            conn.close()

        observation_spaces, action_spaces = [], []
        for conn in self._parent_connections:
            ob_space, ac_space = conn.recv()
            observation_spaces.append(ob_space)
            action_spaces.append(ac_space)
        self.observation_space = SpaceBatch(observation_spaces)
        self.action_space = SpaceBatch(action_spaces)

    @property
    def nenvs(self):
        return self._nenvs

    def _check_actions(self, actions):
        if len(actions) != self._nenvs:
            raise ValueError(
                "number of actions is not equal to number of envs: "
                "len(actions)={}, nenvs={}".format(len(actions), self._nenvs)
            )

    def step(self, actions):
        self._check_actions(actions)
        for conn, a in zip(self._parent_connections, actions):
            conn.send(("step", a))
        results = [conn.recv() for conn in self._parent_connections]
        obs, rews, terminated, truncated, infos = zip(*results)
        return (
            np.stack(obs),
            np.stack(rews),
            np.stack(terminated),
            np.stack(truncated),
            list(infos),
        )

    def reset(self, **kwargs):
        for env_idx, conn in enumerate(self._parent_connections):
            conn.send(("reset", self._seeds[env_idx]))
        results = [conn.recv() for conn in self._parent_connections]
        observations, infos = zip(*results)
        return np.stack(observations), list(infos)

    def close(self):
        if self._closed:
            return
        for conn in self._parent_connections:
            conn.send(("close", None))
        for p in self._processes:
            p.join()
        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()
