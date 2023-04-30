import cv2
from collections import deque
import numpy as np
from core.game import Game
from core.utils import arr_to_str


class AtariWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True):
        """

        :param env: instance of gym environment
        :param k: no. of observations to stack
        """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.uint8)
        # observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA).astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)
        # observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA).astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()
