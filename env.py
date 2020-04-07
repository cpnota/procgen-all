import gym
import torch
import numpy as np
import gym
import cv2
from all.environments import GymEnvironment, State
cv2.ocl.setUseOpenCL(False)


class ProcgenAtariEnv(GymEnvironment):
    def __init__(self, name, *args, **kwargs):
        # need these for duplication
        self._args = args
        self._kwargs = kwargs
        # construct the environment
        env = gym.make('procgen:procgen-{}-v0'.format(name), distribution_mode="easy")
        # apply a subset of wrappers
        env = WarpFrame(env)
        # initialize
        super().__init__(env, *args, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name

    def duplicate(self, n):
        return [
            ProcgenAtariEnv(self._name, *self._args, **self._kwargs) for _ in range(n)
        ]

    def render(self, *args, **kwargs):
        try:
            super().render(*args, **kwargs)
        except:
            return

    def _make_state(self, raw, done, info=None):
        if info is None:
            info = {"life_lost": False}
        elif not "life_lost" in info:
            info["life_lost"] = False
        return State(
            torch.from_numpy(
                np.moveaxis(np.array(raw, dtype=self.state_space.dtype), -1, 0)
            )
            .unsqueeze(0)
            .to(self._device),
            self._done_mask if done else self._not_done_mask,
            [info],
        )

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        '''
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        '''
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs
