from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
from .walker import Walker, ORIG_NAMES
import copy

OBS_TO_CORRECT = [x for x in ORIG_NAMES if (x.endswith('_x') or x.endswith('_y')) and 'pelvis' not in x and 'obst' not in x and 'mass' not in x]


class ObservationFilter:
    MAX_PAST = 8
    MAX_TIMESTEPS = 1000
    def __init__(self, order=2):
        self.arr = []
        self.timesteps = []
        self.t = -1
        self.order = order
        self._extrapolated = np.zeros(ObservationFilter.MAX_TIMESTEPS + 1)

    def step(self, val):
        self.t += 1
        if len(self.arr) == 0 or val != self.arr[-1]:
            self.arr.append(val)
            self.timesteps.append(self.t)

        start_t = max(0, self.t - ObservationFilter.MAX_PAST)
        self._extrapolated[start_t:self.t+1] = self._extrapolate()
        return self._extrapolated[self.t]

    @property
    def extrapolated(self):
        return self._extrapolated[:self.t+1]

    def _extrapolate(self):
        if len(self.arr) <= self.order:
            return np.pad(self.arr, mode='edge', pad_width=(0, self.t - len(self.arr) + 1))

        start_t = max(0, self.t - ObservationFilter.MAX_PAST)
        last_x = ObservationFilter.MAX_PAST + 2
        s = InterpolatedUnivariateSpline(self.timesteps[-last_x:], self.arr[-last_x:], k=self.order)
        return s(np.arange(start_t, self.t + 1, 1))


class NoFilter:
    def __init__(self):
        self._extrapolated = np.zeros(1001)
        self.t = -1

    def step(self, val):
        self.t += 1
        self._extrapolated[self.t] = val
        return val

    @property
    def extrapolated(self):
        return self._extrapolated[:self.t+1]


class VectorObservationFilter:
    def __init__(self, order=2):
        self.order = order
        self.reset()

    def step(self, obs):
        filtered_obs = copy.deepcopy(obs)
        for obs_name, f in self.filters.items():
            obs_idx = ORIG_NAMES.index(obs_name)
            filtered_obs[obs_idx] = f.step(obs[obs_idx])
        return copy.deepcopy(filtered_obs)

    def reset(self):
        self.filters = {name: ObservationFilter(order=self.order) if name in OBS_TO_CORRECT else NoFilter() for name in ORIG_NAMES}

