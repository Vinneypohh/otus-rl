"""
Батч сред: EnvBatch (последовательно) и ParallelEnvBatch (процессы).
По мотивам Practical RL env_batch.
"""

from multiprocessing import Pipe, Process

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Space


class SpaceBatch(Space):
    def __init__(self, spaces):
        first_type = type(spaces[0])
        first_shape = spaces[0].shape
        first_dtype = spaces[0].dtype
        for space in spaces:
            if not isinstance(space, first_type) or first_shape != space.shape or first_dtype != space.dtype:
                raise ValueError("spaces must match")
        self.spaces = spaces
        super().__init__(shape=first_shape, dtype=first_dtype)

    def sample(self):
        return np.stack([space.sample() for space in self.spaces])

    def __getattr__(self, attr):
        return getattr(self.spaces[0], attr)


class EnvBatch(Env):
    def __init__(self, make_env, nenvs):
        self._envs = [make_env() for _ in range(nenvs)]
        self._nenvs = nenvs
        self.observation_space = SpaceBatch([e.observation_space for e in self._envs])
        self.action_space = SpaceBatch([e.action_space for e in self._envs])

    @property
    def nenvs(self):
        return self._nenvs

    def _check_actions(self, actions):
        if len(actions) != self._nenvs:
            raise ValueError("len(actions) != nenvs")

    def step(self, actions):
        self._check_actions(actions)
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for env, a in zip(self._envs, actions):
            o, r, term, trunc, info = env.step(a)
            if term or trunc:
                o, info = env.reset()
            obs_list.append(o)
            rew_list.append(r)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.stack(rew_list),
            np.stack(term_list),
            np.stack(trunc_list),
            info_list,
        )

    def reset(self, **kwargs):
        obs_list, info_list = [], []
        for env in self._envs:
            o, info = env.reset(**kwargs)
            obs_list.append(o)
            info_list.append(info)
        return np.stack(obs_list), info_list

    def close(self):
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


def _worker(parent_conn, worker_conn, make_env, send_spaces=True):
    parent_conn.close()
    env = make_env()
    if send_spaces:
        worker_conn.send((env.observation_space, env.action_space))
    while True:
        cmd, data = worker_conn.recv()
        if cmd == "step":
            o, r, term, trunc, info = env.step(data)
            if term or trunc:
                o, info = env.reset()
            worker_conn.send((o, r, term, trunc, info))
        elif cmd == "reset":
            o, info = env.reset(seed=data)
            worker_conn.send((o, info))
        elif cmd == "close":
            env.close()
            worker_conn.close()
            break
        else:
            raise NotImplementedError(cmd)


class ParallelEnvBatch(Env):
    def __init__(self, make_env, nenvs, seeds=None):
        self._nenvs = nenvs
        self._seeds = seeds if seeds is not None else list(range(nenvs))
        if len(self._seeds) != nenvs:
            self._seeds = (self._seeds * (nenvs // len(self._seeds) + 1))[:nenvs]
        self._parent_conns, self._worker_conns = zip(*[Pipe() for _ in range(nenvs)])
        self._processes = [
            Process(target=_worker, args=(pc, wc, make_env), daemon=True)
            for pc, wc in zip(self._parent_conns, self._worker_conns)
        ]
        for p in self._processes:
            p.start()
        self._closed = False
        for c in self._worker_conns:
            c.close()
        obs_spaces, ac_spaces = [], []
        for c in self._parent_conns:
            ob, ac = c.recv()
            obs_spaces.append(ob)
            ac_spaces.append(ac)
        self.observation_space = SpaceBatch(obs_spaces)
        self.action_space = SpaceBatch(ac_spaces)

    @property
    def nenvs(self):
        return self._nenvs

    def _check_actions(self, actions):
        if len(actions) != self._nenvs:
            raise ValueError("len(actions) != nenvs")

    def step(self, actions):
        self._check_actions(actions)
        for c, a in zip(self._parent_conns, actions):
            c.send(("step", a))
        results = [c.recv() for c in self._parent_conns]
        o, r, term, trunc, info = zip(*results)
        return np.stack(o), np.stack(r), np.stack(term), np.stack(trunc), list(info)

    def reset(self, **kwargs):
        for i, c in enumerate(self._parent_conns):
            c.send(("reset", self._seeds[i]))
        results = [c.recv() for c in self._parent_conns]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)

    def close(self):
        if self._closed:
            return
        for c in self._parent_conns:
            c.send(("close", None))
        for p in self._processes:
            p.join()
        self._closed = True

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.close()
