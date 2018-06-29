from collections import OrderedDict
from collections import namedtuple, deque

import gym.spaces
import numpy as np
import random
import atexit
import socket
import h5py
from datetime import datetime
import time
from gym.spaces import Box
from gym.utils import seeding
from .muscles import comp_activation, INIT_ACTIVATION

import os
from os import path

Spec = namedtuple('Spec', ['timestep_limit', 'id'])

ORIG_NAMES = [
    'pelvis1_rot',   'pelvis1_x',    'pelvis1_y',
    'pelvis1_rot_v', 'pelvis1_x_v',  'pelvis1_y_v',
    'hip_r_rot',     'knee_r_rot',   'ankle_r_rot',   'hip_l_rot',   'knee_l_rot',   'ankle_l_rot',
    'hip_r_rot_v',   'knee_r_rot_v', 'ankle_r_rot_v', 'hip_l_rot_v', 'knee_l_rot_v', 'ankle_l_rot_v',
    'mass_x',     'mass_y',
    'mass_x_v',   'mass_y_v',
    'head_x',     'head_y',
    'pelvis_x',   'pelvis_y',
    'torso_x',    'torso_y',
    'toe_l_x',    'toe_l_y',
    'toe_r_x',    'toe_r_y',
    'talus_l_x',  'talus_l_y',
    'talus_r_x',  'talus_r_y',
    'strength_l', 'strength_r',
    'obst_x', 'obst_y', 'obst_r'
]

ORIG_SYMMETRIES = [
    ('hip_r_rot', 'hip_l_rot'),
    ('knee_r_rot', 'knee_l_rot'),
    ('ankle_r_rot', 'ankle_l_rot'),
    ('toe_l_x', 'toe_r_x'),
    ('toe_l_y', 'toe_r_y'),
    ('talus_l_x', 'talus_r_x'),
    ('talus_l_y', 'talus_r_y'),
    ('strength_l', 'strength_r')
]

ORIG_SYMMETRIC_IDS = list(range(len(ORIG_NAMES)))
for (v1, v2) in ORIG_SYMMETRIES:
    i1 = ORIG_NAMES.index(v1)
    i2 = ORIG_NAMES.index(v2)
    ORIG_SYMMETRIC_IDS[i1], ORIG_SYMMETRIC_IDS[i2] = ORIG_SYMMETRIC_IDS[i2], ORIG_SYMMETRIC_IDS[i1]

NEW2_NAMES = [
    'pelvis_rot',   'pelvis_rot_v', 'pelvis_x_v',  'pelvis_y_v', 'pelvis_y',
    'hip_r_rot',    'knee_r_rot',   'ankle_r_rot',   'hip_l_rot',   'knee_l_rot',   'ankle_l_rot',
    'hip_r_rot_v',  'knee_r_rot_v', 'ankle_r_rot_v', 'hip_l_rot_v', 'knee_l_rot_v', 'ankle_l_rot_v',
    'mass_y', 'mass_x_v', 'mass_y_v',
    'head_y', 'torso_y',
    'toe_l_y', 'toe_r_y',
    'talus_l_y', 'talus_r_y',
    'strength_l',   'strength_r',
    'mass_relx',
    'head_relx',
    'torso_relx',
    'toe_l_relx',
    'toe_r_relx',
    'talus_l_relx',
    'talus_r_relx',
    # 'next_obst_relx', 'next_obst_y', 'next_obst_r',  # Do we need that ??
    # 'prev_obst_relx', 'prev_obst_y', 'prev_obst_r',  # ??
    'talus_l_talus_r_dx', 'talus_r_talus_l_dx',
    'talus_l_talus_r_dy', 'talus_r_talus_l_dy',
    'toe_r_beg_obst_dx', 'toe_l_beg_obst_dx',
    'toe_r_beg_obst_dy', 'toe_l_beg_obst_dy',
    'talus_r_end_obst_dx', 'talus_l_end_obst_dx',
    'talus_r_end_obst_dy', 'talus_l_end_obst_dy'
]

# TODO: NEW2_SYMMETRIES

class SubmitRunEnv:
    def __init__(self, visualize):
        from osim.http.client import Client
        remote_base = "http://grader.crowdai.org:1729"
        self.crowdai_token = "token"
        self.client = Client(remote_base)
        self.first = True
        self.visualize = visualize
        self.action_space = Box(low=0, high=1, shape=[18])
        self.observation_space = Box(low=-3, high=+3, shape=[41])
        self.t = 0
        self.score = 0

    # noinspection PyUnusedLocal
    def reset(self, difficulty, seed):
        self.t = 0
        self.score = 0
        if self.first:
            self.first = False
            return self.client.env_create(self.crowdai_token)
        else:
            obs = self.client.env_reset()  # Might be none
            if obs is None:
                self.client.submit()
                print('SUBMITTED')
                import sys
                sys.exit(0)
            return obs

    def step(self, action):
        obs, rew, done, info = self.client.env_step(list(action), self.visualize)
        self.t += 1
        self.score += rew
        print(f't={self.t:4d} s={self.score:5.2f} speed={100*self.score/self.t:5.2f}')
        import sys
        sys.stdout.flush()
        return obs, rew, done, info

    def close(self):
        pass

    def close_(self):
        pass


def obs_to_dict(orig_obs):
    assert len(orig_obs) == len(ORIG_NAMES)
    return OrderedDict(zip(ORIG_NAMES, orig_obs))


class Walker(gym.Wrapper):
    """ Wraps env (a proper gym env) based on RunEnv. Transforms observations. """
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    OBST_INFTY = 10.

    def __init__(self, env, shaping_mode=None, obstacle_hack=True,
                 transform_inputs=None, max_steps=None, memory_size=1, swap_legs_mode=None, filter_obs=False, add_time=False,
                 fall_penalty=False, fall_penalty_value=2, print_action=False, new8_fix=False, pause=False, noisy_obstacles=False, noisy_obstacles2=False,
                 noisy_fix=False):
        super(Walker, self).__init__(env)

        self.old_obst = (None, None, None, None)
        self.pause = pause
        self.print_action = print_action
        self.new8_fix = new8_fix
        self.filter_obs = filter_obs
        self.fall_penalty = fall_penalty
        self.fall_penalty_value = fall_penalty_value
        self.noisy_obstacles = noisy_obstacles
        self.noisy_obstacles2 = noisy_obstacles2
        self.noisy_fix = noisy_fix
        if filter_obs:
            self.env = FilteredWalker(self.env)

        if add_time:
            assert transform_inputs in ['new_6', 'new_7', 'new_8', 'new_9', 'new_a', 'new_8b']
        if memory_size > 1:
            assert transform_inputs in ['new_3', 'new_4', 'new_5', 'new_6', 'new_7', 'new_8', 'new_9', 'new_a', 'new_8b']
        if transform_inputs in ['new_4', 'new_5']:
            print("WARNING: new_4 and new_5 have bugs")
        self.shaping_mode = shaping_mode
        self.add_time = add_time

        self.action_space = self.env.action_space
        self.transform_inputs = transform_inputs
        self.obstacle_hack = obstacle_hack
        self.max_steps = max_steps
        self.memory_size = memory_size
        self.observation_space = self.get_observation_space()
        self.swap_legs_mode = swap_legs_mode
        self._spec = Spec(timestep_limit=max_steps if max_steps is not None else 1000, id='Walker')
        self.obs_names = [''] * self.observation_space.low.size
        self._seed()

    def _seed(self, seed=None):
        return self.env.seed(seed)

    # noinspection PyMethodMayBeStatic
    def get_observation_space(self):
        if self.transform_inputs == 'relative_x':
            space = Box(low=-3, high=+3, shape=[41 + 18])
        elif self.transform_inputs in [None, 'relative_x', 'relative_x2']:
            space = self.env.observation_space
            space.low[ORIG_NAMES.index('obst_x')], space.high[ORIG_NAMES.index('obst_y')] = 0, 1  # Actually, it might be larger than 1, I believe
        elif self.transform_inputs == 'new_1':
            space = Box(low=-3, high=+3, shape=[46 + 18])
        elif self.transform_inputs == 'new_2':
            space = Box(low=-3, high=+3, shape=[len(NEW2_NAMES)])
        elif self.transform_inputs == 'new_3':
            space = Box(low=-3, high=+3, shape=[27 + self.memory_size * 41])
        elif self.transform_inputs == 'new_4':
            space = Box(low=-3, high=+3, shape=[108])
        elif self.transform_inputs == 'new_5':
            space = Box(low=-3, high=+3, shape=[40 + self.memory_size * 33])
        elif self.transform_inputs == 'new_6':
            space = Box(low=-3, high=+3, shape=[40 + self.memory_size * 33 + (2 if self.add_time else 0)])
        elif self.transform_inputs == 'new_7':
            space = Box(low=-3, high=+3, shape=[40 + self.memory_size * 51 + (2 if self.add_time else 0)])
        elif self.transform_inputs == 'new_8':
            space = Box(low=-3, high=+3, shape=[22 + self.memory_size * 51 + (2 if self.add_time else 0)])
        elif self.transform_inputs == 'new_8b':
            space = Box(low=-3, high=+3, shape=[22 + self.memory_size * 51 + (2 if self.add_time else 0)])
        elif self.transform_inputs == 'new_9':
            space = Box(low=-3, high=+3, shape=[48 + self.memory_size * 33 + (1 if self.add_time else 0)])
        elif self.transform_inputs == 'new_a':
            space = Box(low=-3, high=+3, shape=[90 + (self.memory_size-1) * 47 + (1 if self.add_time else 0)])
        else:
            assert False
        return space

    def _reset(self):
        obs = self.env.reset()
        if isinstance(obs, bool) and obs == False:
            return obs

        self.recent_obs = deque([obs_to_dict(np.zeros_like(self.env.observation_space.low))] * self.memory_size, self.memory_size)
        self.recent_actions = deque([np.zeros_like(self.action_space.low)] * self.memory_size, self.memory_size)
        self.obstacles = []

        self.recent_obs.appendleft(obs_to_dict(obs))
        if not self.transform_inputs == 'new_3':  # Backwards compatibility
            for i in range(self.memory_size - 1):  # Fill the memory with the initial observation
                self.recent_obs.appendleft(obs_to_dict(obs))

        self.last_maximum_step_distance = 0.

        self.muscle_activations = np.full(self.action_space.low.shape, INIT_ACTIVATION)

        self.recent_muscle_activations = deque(maxlen=self.memory_size)
        for i in range(self.memory_size):  # Fill the memory with the initial activations
            self.recent_muscle_activations.appendleft(np.full(self.action_space.low.shape, INIT_ACTIVATION))

        self.t = 0
        self.total_orig_reward = 0
        return self.get_transformed_obs()

    def swap_legs(self, obs_dict):
        assert self.transform_inputs in ['new_3', 'new_4', 'new_5', 'new_6', 'new_7', 'new_8', 'new_8b', 'new_9', 'new_a']  # It might also work for others, but I did not check

        def get_left(key):
            return key.replace('_r_', '_l_')

        def swap(a, b):
            obs_dict[a], obs_dict[b] = obs_dict[b], obs_dict[a]

        keys = obs_dict.keys()
        for key in keys:
            if '_r_' in key:
                swap(key, get_left(key))
        swap('strength_l', 'strength_r')

    def get_transformed_obs(self):
        o = self.recent_obs[0].copy()
        if self.transform_inputs == 'relative_x':
            # Toes relative to talus
            o['toe_r_x'] -= o['talus_r_x']
            o['toe_l_x'] -= o['talus_l_x']

            # Head relative to torso
            o['head_x'] -= o['torso_x']

            # Mass, Torso, Talus and obstacles relative to Pelvis
            o['mass_x'] -= o['pelvis1_x']
            o['torso_x'] -= o['pelvis1_x']
            o['talus_l_x'] -= o['pelvis1_x']
            o['talus_r_x'] -= o['pelvis1_x']
            o['obst_x'] -= o['pelvis1_x']
            o['pelvis_x'] -= o['pelvis1_x']

            for leg in ['r', 'l']:
                for a in range(9):
                    o['muscle_activation_{}_{}'.format(leg, a)] = self.muscle_activations[a]
        elif self.transform_inputs == 'relative_x2':
            # Toes relative to talus
            o['toe_r_x'] -= o['talus_r_x']
            o['toe_l_x'] -= o['talus_l_x']

            # Head relative to torso
            o['head_x'] -= o['torso_x']

            # Mass, Torso, Talus and obstacles relative to Pelvis
            o['mass_x'] -= o['pelvis1_x']
            o['torso_x'] -= o['pelvis1_x']
            o['talus_l_x'] -= o['pelvis1_x']
            o['talus_r_x'] -= o['pelvis1_x']
            o['obst_x'] -= o['pelvis1_x']
            o['pelvis_x'] -= o['pelvis1_x']

            if o['obst_x'] == 100:
                o['obst_x'] = Walker.OBST_INFTY    # 100 is devastating in transfer learning after obstacles
            else:
                o['obst_x'] -= o['pelvis1_x']

            o['pelvis1_x'] = 0

        if self.transform_inputs in [None, 'relative_x']:
            if o['obst_x'] == 100:
                o['obst_x'] = Walker.OBST_INFTY    # 100 is devastating in transfer learning after obstacles

        if self.transform_inputs in [None, 'relative_x', 'relative_x2']:
            self.obs_names[:] = list(o.keys())
            return np.asarray(list(o.values()))

        if self.transform_inputs == 'new_1':
            assert self.obstacle_hack
            obst_is_inf = 1 if o['obst_x'] == 100 else 0
            new_obs = \
                [o[x] for x in ['mass_x_v', 'mass_y_v', 'pelvis1_x_v', 'pelvis1_y_v']] + \
                [o[x] for x in ['pelvis1_y', 'mass_y', 'head_y', 'torso_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']] + \
                [np.sin(o[x]) for x in ['pelvis1_rot', 'hip_r_rot', 'knee_r_rot', 'ankle_r_rot', 'hip_l_rot', 'knee_l_rot', 'ankle_l_rot']] + \
                [np.cos(o[x]) for x in ['pelvis1_rot', 'hip_r_rot', 'knee_r_rot', 'ankle_r_rot', 'hip_l_rot', 'knee_l_rot', 'ankle_l_rot']] + \
                [o[x] / 10. for x in ['pelvis1_rot_v', 'hip_r_rot_v', 'knee_r_rot_v', 'ankle_r_rot_v', 'hip_l_rot_v', 'knee_l_rot_v', 'ankle_l_rot_v']] + \
                [o[x] - o['pelvis_x'] for x in ['mass_x', 'head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x']] + \
                [
                    o['strength_l'], o['strength_r'],
                    obst_is_inf,
                    1 if obst_is_inf else o['obst_x'] - o['pelvis_x'],
                    o['obst_y'],
                    o['obst_r']
                ] + \
                list(self.recent_actions[0])
            assert len(new_obs) == len(self.observation_space.low)
            self.obs_names[:] = list(o.keys())
            return np.asarray(new_obs)
        elif self.transform_inputs == 'new_2':
            orig = self.recent_obs[0]

            o = OrderedDict(zip(NEW2_NAMES, [None for _ in range(len(NEW2_NAMES))]))
            for key in ['pelvis_rot',   'pelvis_rot_v', 'pelvis_x_v',  'pelvis_y_v', 'pelvis_y',
                        'hip_r_rot', 'knee_r_rot',   'ankle_r_rot',   'hip_l_rot',   'knee_l_rot',   'ankle_l_rot',
                        'hip_r_rot_v', 'knee_r_rot_v', 'ankle_r_rot_v', 'hip_l_rot_v', 'knee_l_rot_v', 'ankle_l_rot_v']:
                orig_key = key
                if orig_key == 'pelvis_rot': orig_key = 'pelvis1_rot'
                if orig_key == 'pelvis_rot_v': orig_key = 'pelvis1_rot_v'
                if orig_key == 'pelvis_x_v': orig_key = 'pelvis1_x_v'
                if orig_key == 'pelvis_y_v': orig_key = 'pelvis1_y_v'
                o[key] = orig[orig_key]

            for key in ['mass_y', 'mass_x_v', 'mass_y_v', 'head_y', 'torso_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y',
                        'strength_l', 'strength_r']:
                o[key] = orig[key]
            o['mass_relx'] = orig['mass_x'] - orig['pelvis_x']
            o['head_relx'] = orig['head_x'] - orig['pelvis_x']
            o['torso_relx'] = orig['torso_x'] - orig['pelvis_x']
            o['toe_l_relx'] = orig['toe_l_x'] - orig['talus_l_x']
            o['toe_r_relx'] = orig['toe_r_x'] - orig['talus_r_x']
            o['talus_l_relx'] = orig['talus_l_x'] - orig['pelvis_x']
            o['talus_r_relx'] = orig['talus_r_x'] - orig['pelvis_x']

            o['talus_l_talus_r_dx'] = orig['talus_l_x'] - orig['talus_r_x']
            o['talus_r_talus_l_dx'] = orig['talus_r_x'] - orig['talus_l_x']

            o['talus_l_talus_r_dy'] = orig['talus_l_y'] - orig['talus_r_y']
            o['talus_r_talus_l_dy'] = orig['talus_r_y'] - orig['talus_l_y']

            o['toe_r_beg_obst_dx'] = self.get_dist_next_obst_for('toe_r', 'beg')[0]
            o['toe_l_beg_obst_dx'] = self.get_dist_next_obst_for('toe_l', 'beg')[0]

            o['toe_r_beg_obst_dy'] = self.get_dist_next_obst_for('toe_r', 'beg')[1]
            o['toe_l_beg_obst_dy'] = self.get_dist_next_obst_for('toe_l', 'beg')[1]

            o['talus_r_end_obst_dx'] = self.get_dist_next_obst_for('talus_r', 'end')[0]
            o['talus_l_end_obst_dx'] = self.get_dist_next_obst_for('talus_l', 'end')[0]

            o['talus_r_end_obst_dy'] = self.get_dist_next_obst_for('talus_r', 'end')[1]
            o['talus_l_end_obst_dy'] = self.get_dist_next_obst_for('talus_l', 'end')[1]

            assert all([v is not None for v in o.values()])
            assert len(o) == len(NEW2_NAMES)

            self.obs_names[:] = list(o.keys())
            return np.asarray(list(o.values()))
        elif self.transform_inputs == 'new_3':
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            # Need memory (velocities and things without explicit velocities)
            for key in ['pelvis_rot_v', 'pelvis_x_v', 'pelvis_y_v',
                        'hip_r_rot_v', 'knee_r_rot_v', 'ankle_r_rot_v', 'hip_l_rot_v', 'knee_l_rot_v', 'ankle_l_rot_v',
                        'mass_x_v', 'mass_y_v', 'torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                for mem in range(self.memory_size):
                    o['{}_{}'.format(mem, key)] = self.recent_obs[mem][fix_key(key)]

            # Do not need memory (have explicit velocities)
            for key in ['strength_l', 'strength_r', 'pelvis_rot', 'pelvis_y', 'hip_r_rot', 'hip_l_rot',
                        'knee_r_rot', 'knee_l_rot', 'ankle_r_rot', 'ankle_l_rot', 'mass_y',
                        'obst_x', 'obst_y', 'obst_r']:
                o[key] = self.recent_obs[0][fix_key(key)]

            absolute = self.recent_obs[0]['pelvis_x']

            # Relative, but does not need memory (has velocity)
            for key in ['mass']:
                o['{}_relx'.format(key)] = self.recent_obs[0]['{}_x'.format(key)] - absolute

            # Relative, needs memory
            for key in ['head', 'torso', 'toe_l', 'toe_r', 'talus_l', 'talus_r']:
                for mem in range(self.memory_size):
                    o['{}_{}_relx'.format(mem, key)] = self.recent_obs[mem]['{}_x'.format(key)] - absolute

            o['talus_l_talus_dx'] = self.recent_obs[0]['talus_l_x'] - self.recent_obs[0]['talus_r_x']
            o['talus_r_talus_dx'] = self.recent_obs[0]['talus_r_x'] - self.recent_obs[0]['talus_l_x']

            o['talus_l_talus_dy'] = self.recent_obs[0]['talus_l_y'] - self.recent_obs[0]['talus_r_y']
            o['talus_r_talus_dy'] = self.recent_obs[0]['talus_r_y'] - self.recent_obs[0]['talus_l_y']

            o['toe_r_beg_obst_dx'] = self.get_dist_next_obst_for('toe_r', 'beg')[0]
            o['toe_l_beg_obst_dx'] = self.get_dist_next_obst_for('toe_l', 'beg')[0]

            o['toe_r_beg_obst_dy'] = self.get_dist_next_obst_for('toe_r', 'beg')[1]
            o['toe_l_beg_obst_dy'] = self.get_dist_next_obst_for('toe_l', 'beg')[1]

            o['talus_r_end_obst_dx'] = self.get_dist_next_obst_for('talus_r', 'end')[0]
            o['talus_l_end_obst_dx'] = self.get_dist_next_obst_for('talus_l', 'end')[0]

            o['talus_r_end_obst_dy'] = self.get_dist_next_obst_for('talus_r', 'end')[1]
            o['talus_l_end_obst_dy'] = self.get_dist_next_obst_for('talus_l', 'end')[1]

            for mem in range(self.memory_size):
                for a in range(len(self.action_space.low)):
                    o['{}_a{:02d}'.format(mem, a)] = self.recent_actions[mem][a]

            if self.needs_to_swap():
                self.swap_legs(o)
                for mem in range(self.memory_size):
                    for a in range(9):
                        o['{}_a{:02d}'.format(mem, a)], o['{}_a{:02d}'.format(mem, 9+a)] = o['{}_a{:02d}'.format(mem, 9+a)], o['{}_a{:02d}'.format(mem, a)]

            self.obs_names[:] = list(o.keys())
            return np.asarray(list(o.values()))

        elif self.transform_inputs == 'new_4':
            assert self.memory_size == 3
            mem = self.recent_obs
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            # Do not need deltas (e.g., have explicit velocities)
            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot', 'ankle_l_rot',
                        'pelvis_y', 'mass_y', 'strength_l', 'strength_r']:
                o[key] = mem[0][fix_key(key)]

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = mem[0][key]
                o[key] = v if v < 100 else Walker.OBST_INFTY
            o['no_obstacles'] = 1. if mem[0]['obst_x'] == 100.0 else 0.

            # Velocities need deltas (accelerations)
            for key in ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v', 'ankle_l_rot_v',
                        'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v']:
                key = fix_key(key)
                o[key] = mem[0][key]
                o['d_' + key] = mem[0][key] - mem[1][key]

            # Positions need deltas (velocities) and deltas deltas (accelerations)
            for key in ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[0][key]
                o['d_' + key] = mem[0][key] - mem[1][key]
                o['dd_' + key] = mem[0][key] - 2 * mem[1][key] + mem[2][key]

            absolute = self.recent_obs[0]['pelvis_x']

            # Relative, but has explicit velocity
            for key in ['mass_x']:
                o['rel_' + key] = mem[0][key] - absolute

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x']:
                o['rel_' + key] = mem[0][key] - absolute
                o['d_{}'.format(key)] = mem[0][key] - mem[1][key]  # velocities
                o['dd_{}'.format(key)] = mem[0][key] - 2 * mem[1][key] + mem[2][key]  # acceleration

            o['dist_talus_l_talus_x'] = mem[0]['talus_l_x'] - mem[0]['talus_r_x']
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem[0]['talus_l_y'] - mem[0]['talus_r_y']
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem[0]['talus_l_x'] - mem[0]['talus_r_x'] - (mem[1]['talus_l_x'] - mem[1]['talus_r_x'])
            o['d_dist_talus_r_talus_x'] = mem[0]['talus_r_x'] - mem[0]['talus_l_x'] - (mem[1]['talus_r_x'] - mem[1]['talus_l_x'])

            o['d_dist_talus_l_talus_y'] = mem[0]['talus_l_y'] - mem[0]['talus_r_y'] - (mem[1]['talus_l_y'] - mem[1]['talus_r_y'])
            o['d_dist_talus_r_talus_y'] = mem[0]['talus_r_y'] - mem[0]['talus_l_y'] - (mem[1]['talus_r_y'] - mem[1]['talus_l_y'])

            o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for('toe_r', 'beg')[0]
            o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for('toe_l', 'beg')[0]

            o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for('toe_r', 'beg')[1]
            o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for('toe_l', 'beg')[1]

            o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for('talus_r', 'end')[0]
            o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for('talus_l', 'end')[0]

            o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for('talus_r', 'end')[1]
            o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for('talus_l', 'end')[1]

            for leg in ['r', 'l']:
                for a in range(9):
                    o['muscle_activation_{}_{}'.format(leg, a)] = self.muscle_activations[a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())
            return np.asarray(list(o.values()))
        elif self.transform_inputs == 'new_5':
            assert self.filter_obs
            NEW5_INFTY = 2
            mem = self.env.filtered_memory
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            for key in ['strength_l', 'strength_r']:
                o[key] = mem[key][0]

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = mem[key][0]
                o[key] = v if v < 100 else NEW5_INFTY
            o['no_obstacles'] = 1. if mem['obst_x'][0] == 100.0 else 0.

            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot',
                        'ankle_l_rot', 'pelvis_y', 'mass_y'] + \
                       ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v',
                        'ankle_l_rot_v', 'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v'] + \
                       ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[key][0]
                for h in range(self.memory_size - 1):
                    o['d' + str(h) + '_' + key] = mem[key][0] - mem[key][h + 1]

            absolute = mem['pelvis_x'][0]

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x', 'mass_x']:
                o['rel_' + key] = mem[key][0] - absolute
                for h in range(self.memory_size - 1):
                    o['d' + str(h) + '_' + key] = mem[key][0] - mem[key][h + 1]

            o['dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0]
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0]
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0] - (mem['talus_l_x'][1] - mem['talus_r_x'][1])
            o['d_dist_talus_r_talus_x'] = mem['talus_r_x'][0] - mem['talus_l_x'][0] - (mem['talus_r_x'][1] - mem['talus_l_x'][1])

            o['d_dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0] - (mem['talus_l_y'][1] - mem['talus_r_y'][1])
            o['d_dist_talus_r_talus_y'] = mem['talus_r_y'][0] - mem['talus_l_y'][0] - (mem['talus_r_y'][1] - mem['talus_l_y'][1])

            o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW5_INFTY)[0]
            o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW5_INFTY)[0]

            o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW5_INFTY)[1]
            o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW5_INFTY)[1]

            o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for('talus_r', 'end', NEW5_INFTY)[0]
            o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for('talus_l', 'end', NEW5_INFTY)[0]

            o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for('talus_r', 'end', NEW5_INFTY)[1]
            o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for('talus_l', 'end', NEW5_INFTY)[1]

            for leg in ['r', 'l']:
                for a in range(9):
                    o['muscle_activation_{}_{}'.format(leg, a)] = self.muscle_activations[a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())
            return np.asarray(list(o.values()))
        elif self.transform_inputs == 'new_6':
            assert self.filter_obs
            NEW6_INFTY = 3
            mem = self.env.filtered_memory
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            for key in ['strength_l', 'strength_r']:
                o[key] = mem[key][0]

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = mem[key][0]
                o[key] = v if v < 100 else NEW6_INFTY
            o['no_obstacles'] = 1. if mem['obst_x'][0] == 100.0 else 0.

            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot',
                        'ankle_l_rot', 'pelvis_y', 'mass_y'] + \
                       ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v',
                        'ankle_l_rot_v', 'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v'] + \
                       ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[key][0]
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][0] - mem[key][h]

            absolute = mem['pelvis_x'][0]

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x', 'mass_x']:
                o['rel_' + key] = mem[key][0] - absolute
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][0] - mem[key][h]

            o['dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0]
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0]
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0] - (mem['talus_l_x'][1] - mem['talus_r_x'][1])
            o['d_dist_talus_r_talus_x'] = mem['talus_r_x'][0] - mem['talus_l_x'][0] - (mem['talus_r_x'][1] - mem['talus_l_x'][1])

            o['d_dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0] - (mem['talus_l_y'][1] - mem['talus_r_y'][1])
            o['d_dist_talus_r_talus_y'] = mem['talus_r_y'][0] - mem['talus_l_y'][0] - (mem['talus_r_y'][1] - mem['talus_l_y'][1])

            o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[0]
            o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[0]

            o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[1]
            o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[1]

            o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[0]
            o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[0]

            o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[1]
            o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[1]

            if self.add_time:
                o['t'] = self.t
                o['rew'] = self.total_orig_reward

            for leg in ['r', 'l']:
                for a in range(9):
                    leg_a = a if leg == 'r' else a + 9
                    o['muscle_activation_{}_{}'.format(leg, a)] = self.muscle_activations[leg_a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())
            return np.asarray(list(o.values()))
        elif self.transform_inputs == 'new_7':
            assert self.filter_obs
            NEW6_INFTY = 3
            mem = self.env.filtered_memory
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            for key in ['strength_l', 'strength_r']:
                o[key] = mem[key][0]

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = mem[key][0]
                o[key] = v if v < 100 else NEW6_INFTY
            o['no_obstacles'] = 1. if mem['obst_x'][0] == 100.0 else 0.

            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot',
                        'ankle_l_rot', 'pelvis_y', 'mass_y'] + \
                       ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v',
                        'ankle_l_rot_v', 'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v'] + \
                       ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[key][0]
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            absolute = mem['pelvis_x'][0]

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x', 'mass_x']:
                o['rel_' + key] = mem[key][0] - absolute
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            o['dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0]
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0]
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0] - (mem['talus_l_x'][1] - mem['talus_r_x'][1])
            o['d_dist_talus_r_talus_x'] = mem['talus_r_x'][0] - mem['talus_l_x'][0] - (mem['talus_r_x'][1] - mem['talus_l_x'][1])

            o['d_dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0] - (mem['talus_l_y'][1] - mem['talus_r_y'][1])
            o['d_dist_talus_r_talus_y'] = mem['talus_r_y'][0] - mem['talus_l_y'][0] - (mem['talus_r_y'][1] - mem['talus_l_y'][1])

            o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[0]
            o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[0]

            o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[1]
            o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[1]

            o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[0]
            o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[0]

            o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[1]
            o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[1]

            if self.add_time:
                o['t'] = self.t
                o['rew'] = self.total_orig_reward

            for leg in ['r', 'l']:
                for a in range(9):
                    leg_a = a if leg == 'r' else a + 9
                    o['muscle_activation_{}_{}'.format(leg, a)] = self.muscle_activations[leg_a]
            for mem in range(self.memory_size):
                for leg in ['r', 'l']:
                    for a in range(9):
                        leg_a = a if leg == 'r' else a + 9
                        o['actions_{}_{}_{}'.format(mem, leg, a)] = self.recent_actions[mem][leg_a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())

            return np.asarray(list(o.values()))

        elif self.transform_inputs == 'new_8':
            assert self.filter_obs
            NEW6_INFTY = 3
            mem = self.env.filtered_memory
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            for key in ['strength_l', 'strength_r']:
                o[key] = mem[key][0]
                if self.noisy_obstacles:
                    o[key] += random.gauss(0, 0.1)
                if self.noisy_obstacles2:
                    o[key] += random.gauss(0, 0.2)

            assert mem['obst_r'][0] == self.recent_obs[0]['obst_r']

            if self.noisy_obstacles:
                noisy = self.recent_obs[0].copy()
                noisy["obst_x"] += random.gauss(0, 0.02)
                noisy["obst_y"] += random.gauss(0, 0.01)
                noisy["obst_r"] += random.gauss(0, 0.02)
            if self.noisy_obstacles2:
                noisy = self.recent_obs[0].copy()
                noisy["obst_x"] += random.gauss(0, 0.04)
                noisy["obst_y"] += random.gauss(0, 0.02)
                noisy["obst_r"] += random.gauss(0, 0.04)

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = noisy[key] if self.noisy_fix and (self.noisy_obstacles or self.noisy_obstacles2) else mem[key][0]
                o[key] = v if v < 100 else NEW6_INFTY
            o['no_obstacles'] = 1. if mem['obst_x'][0] == 100.0 else 0.

            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot',
                        'ankle_l_rot', 'pelvis_y', 'mass_y'] + \
                    ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v',
                     'ankle_l_rot_v', 'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v'] + \
                    ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[key][0]
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            absolute = mem['pelvis_x'][0]

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x', 'mass_x']:
                o['rel_' + key] = mem[key][0] - absolute
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            o['dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0]
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0]
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0] - (mem['talus_l_x'][1] - mem['talus_r_x'][1])
            o['d_dist_talus_r_talus_x'] = mem['talus_r_x'][0] - mem['talus_l_x'][0] - (mem['talus_r_x'][1] - mem['talus_l_x'][1])

            o['d_dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0] - (mem['talus_l_y'][1] - mem['talus_r_y'][1])
            o['d_dist_talus_r_talus_y'] = mem['talus_r_y'][0] - mem['talus_l_y'][0] - (mem['talus_r_y'][1] - mem['talus_l_y'][1])

            if self.new8_fix:
                old_toe_r, old_toe_l, old_talus_r, old_talus_l = self.old_obst

                if self.noisy_obstacles or self.noisy_obstacles2:
                    o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for_new8b_noisy('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r, noisy)[0]
                    o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for_new8b_noisy('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l, noisy)[0]

                    o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for_new8b_noisy('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r, noisy)[1]
                    o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for_new8b_noisy('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l, noisy)[1]

                    o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for_new8b_noisy('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r, noisy)[0]
                    o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for_new8b_noisy('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l, noisy)[0]

                    o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for_new8b_noisy('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r, noisy)[1]
                    o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for_new8b_noisy('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l, noisy)[1]
                else:
                    o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for_new8b('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r)[0]
                    o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for_new8b('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l)[0]

                    o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for_new8b('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r)[1]
                    o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for_new8b('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l)[1]

                    o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for_new8b('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r)[0]
                    o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for_new8b('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l)[0]

                    o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for_new8b('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r)[1]
                    o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for_new8b('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l)[1]

                self.old_obst = (
                    self.get_dist_next_obst_for_new8b('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r),
                    self.get_dist_next_obst_for_new8b('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l),
                    self.get_dist_next_obst_for_new8b('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r),
                    self.get_dist_next_obst_for_new8b('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l),
                )
            else:
                o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[0]
                o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[0]

                o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[1]
                o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[1]

                o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[0]
                o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[0]

                o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[1]
                o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[1]

            if self.add_time:
                o['t'] = self.t
                o['rew'] = self.total_orig_reward

            for mem in range(self.memory_size):
                for leg in ['r', 'l']:
                    for a in range(9):
                        leg_a = a if leg == 'r' else a + 9
                        o['muscle_activation_{}_{}_{}'.format(mem, leg, a)] = self.recent_muscle_activations[mem][leg_a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())

            #print('obs_names=' + str( self.obs_names))#, compact=True, width=None)

            return np.asarray(list(o.values()))
        elif self.transform_inputs == 'new_8b':
            assert self.filter_obs
            NEW6_INFTY = 10
            mem = self.env.filtered_memory
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            for key in ['strength_l', 'strength_r']:
                o[key] = mem[key][0]

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = mem[key][0]
                o[key] = v if v < 100 else NEW6_INFTY
            o['no_obstacles'] = 1. if mem['obst_x'][0] == 100.0 else 0.

            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot',
                        'ankle_l_rot', 'pelvis_y', 'mass_y'] + \
                    ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v',
                     'ankle_l_rot_v', 'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v'] + \
                    ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[key][0]
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            absolute = mem['pelvis_x'][0]

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x', 'mass_x']:
                o['rel_' + key] = mem[key][0] - absolute
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            o['dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0]
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0]
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0] - (mem['talus_l_x'][1] - mem['talus_r_x'][1])
            o['d_dist_talus_r_talus_x'] = mem['talus_r_x'][0] - mem['talus_l_x'][0] - (mem['talus_r_x'][1] - mem['talus_l_x'][1])

            o['d_dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0] - (mem['talus_l_y'][1] - mem['talus_r_y'][1])
            o['d_dist_talus_r_talus_y'] = mem['talus_r_y'][0] - mem['talus_l_y'][0] - (mem['talus_r_y'][1] - mem['talus_l_y'][1])

            old_toe_r, old_toe_l, old_talus_r, old_talus_l = self.old_obst

            o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for_new8b('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r)[0]
            o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for_new8b('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l)[0]

            o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for_new8b('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r)[1]
            o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for_new8b('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l)[1]

            o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for_new8b('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r)[0]
            o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for_new8b('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l)[0]

            o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for_new8b('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r)[1]
            o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for_new8b('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l)[1]

            self.old_obst = (
                self.get_dist_next_obst_for_new8b('toe_r', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_r),
                self.get_dist_next_obst_for_new8b('toe_l', 'beg', NEW6_INFTY, o['no_obstacles'], old_toe_l),
                self.get_dist_next_obst_for_new8b('talus_r', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_r),
                self.get_dist_next_obst_for_new8b('talus_l', 'end', NEW6_INFTY, o['no_obstacles'], old_talus_l),
            )

            if self.add_time:
                o['t'] = self.t
                o['rew'] = self.total_orig_reward

            for mem in range(self.memory_size):
                for leg in ['r', 'l']:
                    for a in range(9):
                        leg_a = a if leg == 'r' else a + 9
                        o['muscle_activation_{}_{}_{}'.format(mem, leg, a)] = self.recent_muscle_activations[mem][leg_a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())

            #print('obs_names=' + str( self.obs_names))#, compact=True, width=None)

            return np.asarray(list(o.values()))
        elif self.transform_inputs == 'new_9':
            assert self.filter_obs
            NEW6_INFTY = 3
            mem = self.env.filtered_memory
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            for key in ['strength_l', 'strength_r']:
                o[key] = mem[key][0]

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = mem[key][0]
                o[key] = v if v < 100 else NEW6_INFTY
            o['no_obstacles'] = 1. if mem['obst_x'][0] == 100.0 else 0.

            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot',
                        'ankle_l_rot', 'pelvis_y', 'mass_y'] + \
                    ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v',
                     'ankle_l_rot_v', 'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v'] + \
                    ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[key][0]
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            absolute = mem['pelvis_x'][0]

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x', 'mass_x']:
                o['rel_' + key] = mem[key][0] - absolute
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            o['dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0]
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0]
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0] - (mem['talus_l_x'][1] - mem['talus_r_x'][1])
            o['d_dist_talus_r_talus_x'] = mem['talus_r_x'][0] - mem['talus_l_x'][0] - (mem['talus_r_x'][1] - mem['talus_l_x'][1])

            o['d_dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0] - (mem['talus_l_y'][1] - mem['talus_r_y'][1])
            o['d_dist_talus_r_talus_y'] = mem['talus_r_y'][0] - mem['talus_l_y'][0] - (mem['talus_r_y'][1] - mem['talus_l_y'][1])

            o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[0]
            o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[0]

            o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[1]
            o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[1]

            o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[0]
            o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[0]

            o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[1]
            o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[1]

            if self.add_time:
                o['t'] = self.t / 1000

            for key in ['toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:  # y of toes and taluses
                # touch_ind = 1 if new[key] < 0.05 else 0
                o['touch_indicator_' + key] = np.clip(0.05 - mem[key][0] * 10 + 0.5, 0., 1.)
                o['touch_indicator_2_' + key] = np.clip(0.1 - mem[key][0] * 10 + 0.5, 0., 1.)

            for leg in ['r', 'l']:
                for a in range(9):
                    leg_a = a if leg == 'r' else a + 9
                    o['muscle_activation_{}_{}'.format(leg, a)] = self.muscle_activations[leg_a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())

            return np.asarray(list(o.values()))
        elif self.transform_inputs == 'new_a':
            assert self.filter_obs
            NEW6_INFTY = 3
            mem = self.env.filtered_memory
            o = OrderedDict()

            def fix_key(name):
                if name == 'pelvis_rot': return 'pelvis1_rot'
                if name == 'pelvis_rot_v': return 'pelvis1_rot_v'
                if name == 'pelvis_x_v': return 'pelvis1_x_v'
                if name == 'pelvis_y_v': return 'pelvis1_y_v'
                return name

            for key in ['strength_l', 'strength_r']:
                o[key] = mem[key][0]

            # Obstacles
            for key in ['obst_x', 'obst_y', 'obst_r']:
                v = mem[key][0]
                o[key] = v if v < 100 else NEW6_INFTY
            o['no_obstacles'] = 1. if mem['obst_x'][0] == 100.0 else 0.

            for key in ['pelvis_rot', 'hip_r_rot', 'hip_l_rot', 'knee_r_rot', 'knee_l_rot', 'ankle_r_rot',
                        'ankle_l_rot', 'pelvis_y', 'mass_y'] + \
                    ['pelvis_rot_v', 'hip_r_rot_v', 'hip_l_rot_v', 'knee_r_rot_v', 'knee_l_rot_v', 'ankle_r_rot_v',
                     'ankle_l_rot_v', 'pelvis_y_v', 'mass_y_v', 'mass_x_v', 'pelvis_x_v'] + \
                    ['torso_y', 'head_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:
                key = fix_key(key)
                o[key] = mem[key][0]
                for h in range(1, self.memory_size):
                    o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]

            absolute = mem['pelvis_x'][0]
            absolute_y = mem['pelvis_y'][0]
            absolute_x_v = mem[fix_key('pelvis_x_v')][0]
            absolute_y_v = mem[fix_key('pelvis_x_v')][0]

            # Relative that need velocity and acceleration
            for key in ['head_x', 'torso_x', 'toe_l_x', 'toe_r_x', 'talus_l_x', 'talus_r_x', 'mass_x']:
                o['rel_' + key] = mem[key][0] - absolute
                for h in range(1, self.memory_size):
                    v = o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]
                    o['rel_v_' + key + '_' + str(h)] = absolute_x_v - v

            for key in ['head_y', 'torso_y', 'toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y', 'mass_y']:
                o['rel_' + key] = mem[key][0] - absolute_y
                for h in range(1, self.memory_size):
                    v = o['d' + str(h) + '_' + key] = mem[key][h - 1] - mem[key][h]
                    o['rel_v_' + key + '_' + str(h)] = absolute_y_v - v

            for key in ['mass_y_v']:
                o['rel_' + key] = mem[key][0] - absolute_y_v

            for key in ['mass_x_v']:
                o['rel_' + key] = mem[key][0] - absolute_x_v


            o['dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0]
            o['dist_talus_r_talus_x'] = -o['dist_talus_l_talus_x']

            o['dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0]
            o['dist_talus_r_talus_y'] = -o['dist_talus_l_talus_y']

            # Velocities of the distances
            o['d_dist_talus_l_talus_x'] = mem['talus_l_x'][0] - mem['talus_r_x'][0] - (mem['talus_l_x'][1] - mem['talus_r_x'][1])
            o['d_dist_talus_r_talus_x'] = mem['talus_r_x'][0] - mem['talus_l_x'][0] - (mem['talus_r_x'][1] - mem['talus_l_x'][1])

            o['d_dist_talus_l_talus_y'] = mem['talus_l_y'][0] - mem['talus_r_y'][0] - (mem['talus_l_y'][1] - mem['talus_r_y'][1])
            o['d_dist_talus_r_talus_y'] = mem['talus_r_y'][0] - mem['talus_l_y'][0] - (mem['talus_r_y'][1] - mem['talus_l_y'][1])

            o['dist_toe_r_beg_obst_x'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[0]
            o['dist_toe_l_beg_obst_x'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[0]

            o['dist_toe_r_beg_obst_y'] = self.get_dist_next_obst_for('toe_r', 'beg', NEW6_INFTY)[1]
            o['dist_toe_l_beg_obst_y'] = self.get_dist_next_obst_for('toe_l', 'beg', NEW6_INFTY)[1]

            o['dist_talus_r_end_obst_x'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[0]
            o['dist_talus_l_end_obst_x'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[0]

            o['dist_talus_r_end_obst_y'] = self.get_dist_next_obst_for('talus_r', 'end', NEW6_INFTY)[1]
            o['dist_talus_l_end_obst_y'] = self.get_dist_next_obst_for('talus_l', 'end', NEW6_INFTY)[1]

            if self.add_time:
                o['t'] = self.t / 1000

            for key in ['toe_l_y', 'toe_r_y', 'talus_l_y', 'talus_r_y']:  # y of toes and taluses
                # touch_ind = 1 if new[key] < 0.05 else 0
                o['touch_indicator_' + key] = np.clip(0.05 - mem[key][0] * 10 + 0.5, 0., 1.)
                o['touch_indicator_2_' + key] = np.clip(0.1 - mem[key][0] * 10 + 0.5, 0., 1.)

            for leg in ['r', 'l']:
                for a in range(9):
                    leg_a = a if leg == 'r' else a + 9
                    o['muscle_activation_{}_{}'.format(leg, a)] = self.muscle_activations[leg_a]

            if self.needs_to_swap():
                self.swap_legs(o)

            self.obs_names[:] = list(o.keys())

            return np.asarray(list(o.values()))

        assert False

    def update_obstacles(self):
        orig_obs = self.recent_obs[0]
        if orig_obs['obst_x'] == 100.0:
            return

        obst_abs_x = orig_obs['obst_x'] + orig_obs['pelvis_x']
        curr_obs = (obst_abs_x, orig_obs['obst_y'], orig_obs['obst_r'])

        def similar(saved, curr):
            for s, c in zip(saved, curr):
                if abs(s - c) > 1e-4:
                    return False
            return True

        if len(self.obstacles) == 0:
            self.obstacles.append(curr_obs)
        else:
            # TODO Compare all not only the last one
            last = self.obstacles[-1]
            if not similar(last, curr_obs):
                self.obstacles.append(curr_obs)
        self.obstacles.sort(key=lambda o: o[0] - o[2])  # just in case (x - radius)

    def get_dist_next_obst_for(self, name, pos, infty=OBST_INFTY):
        if len(self.obstacles) == 0:
            return infty, infty

        my_x = self.recent_obs[0][name + '_x']
        my_y = self.recent_obs[0][name + '_y']
        for obst_x, obst_y, obst_r in self.obstacles:
            obst_beg_x = obst_x - obst_r
            obst_end_x = obst_x + obst_r
            obst_top = obst_y + obst_r
            if pos == 'beg':
                if my_x < obst_beg_x:
                    dist_x = obst_beg_x - my_x
                    dist_y = obst_top - my_y
                    return dist_x, dist_y
            elif pos == 'end':
                if my_x < obst_end_x:
                    dist_x = obst_end_x - my_x
                    dist_y = obst_top - my_y
                    return dist_x, dist_y
            else:
                assert False
        return infty, infty

    def get_dist_next_obst_for_new8b(self, name, pos, infty, no_obstacles, old):
        if len(self.obstacles) == 0:
            return infty, infty

        my_x = self.recent_obs[0][name + '_x']
        my_y = self.recent_obs[0][name + '_y']
        for obst_x, obst_y, obst_r in self.obstacles:
            obst_beg_x = obst_x - obst_r
            obst_end_x = obst_x + obst_r
            obst_top = obst_y + obst_r
            if pos == 'beg':
                if my_x < obst_beg_x:
                    dist_x = obst_beg_x - my_x
                    dist_y = obst_top - my_y
                    return dist_x, dist_y
            elif pos == 'end':
                if my_x < obst_end_x:
                    dist_x = obst_end_x - my_x
                    dist_y = obst_top - my_y
                    return dist_x, dist_y
            else:
                assert False
        if no_obstacles or old is None:
            return infty, infty
        else:
            return old

    def get_dist_next_obst_for_new8b_noisy(self, name, pos, infty, no_obstacles, old, noisy):
        if len(self.obstacles) == 0:
            return infty, infty

        my_x = noisy[name + '_x']
        my_y = noisy[name + '_y']
        for obst_x, obst_y, obst_r in self.obstacles:
            obst_beg_x = obst_x - obst_r
            obst_end_x = obst_x + obst_r
            obst_top = obst_y + obst_r
            if pos == 'beg':
                if my_x < obst_beg_x:
                    dist_x = obst_beg_x - my_x
                    dist_y = obst_top - my_y
                    return dist_x, dist_y
            elif pos == 'end':
                if my_x < obst_end_x:
                    dist_x = obst_end_x - my_x
                    dist_y = obst_top - my_y
                    return dist_x, dist_y
            else:
                assert False
        if no_obstacles or old is None:
            return infty, infty
        else:
            return old

    def needs_to_swap(self):
        if self.swap_legs_mode is None:
            return False
        m = self.recent_obs
        if self.swap_legs_mode == 'position':
            return m[0]['talus_l_x'] < m[0]['talus_r_x']
        elif self.swap_legs_mode == 'velocity':
            if len(m) > 1:
                return m[0]['talus_l_x'] - m[1]['talus_l_x'] < m[0]['talus_r_x'] - m[1]['talus_r_x']
            else:  # Funny... # TODO: This is wrong since it randomizes twice (every time needs_to_swap is executed)
                return random.random() < 0.5
        else:
            assert False

    def _step(self, action):
        # Actions are (potentially) front-back
        if self.pause and self.t == 0:
            input('hit to return')
        if self.needs_to_swap():
            tmp = action.copy()
            action[:9], action[9:] = tmp[9:], tmp[:9]

        # Now actions are left-right
        actual_action = np.clip(action, self.action_space.low, self.action_space.high)

        # Activations are left-right
        self.muscle_activations = comp_activation(self.muscle_activations, actual_action)
        self.recent_muscle_activations.appendleft(self.muscle_activations)

        previous_obs = self.recent_obs[0]
        obs, reward, done, info = self.env.step(actual_action)
        self.recent_actions.appendleft(actual_action)
        self.recent_obs.appendleft(obs_to_dict(obs))  # recent_obs are left-right
        self.update_obstacles()

        info['original_reward'] = reward
        self.total_orig_reward += reward

        if self.shaping_mode is not None and self.shaping_mode != 'None':
            reward = self.get_shaped_reward(reward, self.recent_obs[0], previous_obs, self.shaping_mode)

        self.t += 1
        done_maxsteps = False
        if self.max_steps is not None and self.t >= self.max_steps:
            done = True
            done_maxsteps = True
        info['done_maxsteps'] = done_maxsteps
        #print(self.t, info['done_maxsteps'])

        if self.fall_penalty:
            if done and not done_maxsteps:
                reward -= self.fall_penalty_value

        if self.print_action:
            print(self.t, 'action', action, 'reward', reward, obs[ORIG_NAMES.index('strength_l')], obs[ORIG_NAMES.index('strength_r')])

        return self.get_transformed_obs(), reward, done, info  # transformation transforms to front-back

    def _render(self, mode='human', close=False):
        self.env.render(mode, close)

    def get_shaped_reward(self, orig_reward, orig_obs, last_orig_obs, mode):
        if mode in ['head_foot_1', 'hf1']:
            foot_l_delta = (orig_obs['talus_l_x'] - last_orig_obs['talus_l_x']) / 2.
            foot_r_delta = (orig_obs['talus_r_x'] - last_orig_obs['talus_r_x']) / 2.
            return orig_reward + foot_l_delta + foot_r_delta + 0.01 * (orig_obs['head_y'] > 1.3)
        elif mode == 'st1':
            fall_penalty = np.maximum(abs((orig_obs['head_x'] - orig_obs['pelvis_x'] + 0.27) / 0.4 - 0.5) - 0.5, 0)
            reward = orig_reward - fall_penalty
            # print(f'{self.t:3d} {reward:+2.4f} {orig_reward:+2.4f} {fall_penalty:+2.4f}')
            return reward
        elif mode == 'long_head_high':
            return orig_reward + 0.01 * (orig_obs['head_y'] > 1.4)
        elif mode == 'long_straight':
            long_head_up_reward = 0.04 * (orig_obs['head_y'] - 1.4)
            pelvis_mass = -0.05 * max(abs(orig_obs['mass_x'] - orig_obs['pelvis_x']) - 0.2, 0)
            return orig_reward + long_head_up_reward + pelvis_mass
        elif mode == 'ls2':
            long_head_up_reward = 0.015 * (orig_obs['head_y'] - 1.4)
            pelvis_mass = -0.025 * max(abs(orig_obs['mass_x'] - orig_obs['pelvis_x']) - 0.2, 0)
            return orig_reward + long_head_up_reward + pelvis_mass
        elif mode == 'hp1':
            head_pelvis_dx = orig_obs['head_x'] - orig_obs['pelvis_x']
            if head_pelvis_dx < 0.46:
                penalty = 0.
            else:
                penalty = 0.2 + (head_pelvis_dx - 0.37)/0.3*0.1
            return orig_reward - penalty
        elif mode == 'head_foot_2':
            pelvis1_delta = (orig_obs['pelvis1_x'] - last_orig_obs['pelvis1_x'])
            foot_l_delta = (orig_obs['talus_l_x'] - last_orig_obs['talus_l_x'])
            foot_r_delta = (orig_obs['talus_r_x'] - last_orig_obs['talus_r_x'])
            head_delta = (orig_obs['head_x'] - last_orig_obs['head_x'])
            return orig_reward - pelvis1_delta + (pelvis1_delta + foot_l_delta + foot_r_delta + head_delta) / 4.
        elif mode == 'head_foot_step':
            pelvis1_delta = (orig_obs['pelvis1_x'] - last_orig_obs['pelvis1_x'])
            foot_l_delta = (orig_obs['talus_l_x'] - last_orig_obs['talus_l_x'])
            foot_r_delta = (orig_obs['talus_r_x'] - last_orig_obs['talus_r_x'])
            head_delta = (orig_obs['head_x'] - last_orig_obs['head_x'])
            talus_order = (last_orig_obs['talus_l_x'] - last_orig_obs['talus_r_x']) * (orig_obs['talus_l_x'] - orig_obs['talus_r_x'])
            talus_order_change = talus_order < 0
            if talus_order_change:
                step_reward = self.last_maximum_step_distance / 2
                self.last_maximum_step_distance = 0
            else:
                step_reward = 0
                self.last_maximum_step_distance = max(self.last_maximum_step_distance, abs(orig_obs['talus_l_x'] - orig_obs['talus_r_x']))
            return orig_reward - pelvis1_delta + (pelvis1_delta + foot_l_delta + foot_r_delta + head_delta) / 4 + step_reward
        elif mode == 'hfs_knee':
            pelvis1_delta = (orig_obs['pelvis1_x'] - last_orig_obs['pelvis1_x'])
            foot_l_delta = (orig_obs['talus_l_x'] - last_orig_obs['talus_l_x'])
            foot_r_delta = (orig_obs['talus_r_x'] - last_orig_obs['talus_r_x'])
            head_delta = (orig_obs['head_x'] - last_orig_obs['head_x'])
            talus_order = (last_orig_obs['talus_l_x'] - last_orig_obs['talus_r_x']) * (orig_obs['talus_l_x'] - orig_obs['talus_r_x'])
            talus_order_change = talus_order < 0
            if talus_order_change:
                step_reward = self.last_maximum_step_distance / 2
                self.last_maximum_step_distance = 0
            else:
                step_reward = 0
                self.last_maximum_step_distance = max(self.last_maximum_step_distance, abs(orig_obs['talus_l_x'] - orig_obs['talus_r_x']))
            knee_reward = -(orig_obs['knee_r_rot'] + orig_obs['knee_l_rot']) / 100  # TODO: Not 100% sure
            return orig_reward - pelvis1_delta + (pelvis1_delta + foot_l_delta + foot_r_delta + head_delta) / 4 + step_reward + knee_reward
        elif mode == 'hf_knee_2':
            pelvis1_delta = (orig_obs['pelvis1_x'] - last_orig_obs['pelvis1_x'])
            foot_l_delta = (orig_obs['talus_l_x'] - last_orig_obs['talus_l_x'])
            foot_r_delta = (orig_obs['talus_r_x'] - last_orig_obs['talus_r_x'])
            head_delta = (orig_obs['head_x'] - last_orig_obs['head_x'])
            knee_reward = -(max(orig_obs['knee_r_rot'], -np.pi/4) + max(orig_obs['knee_l_rot'], -np.pi/4)) / 300
            return orig_reward - pelvis1_delta + (pelvis1_delta + foot_l_delta + foot_r_delta + head_delta) / 4 + knee_reward
        elif mode == 'addknee':
            knee_reward = (min(-orig_obs['knee_r_rot'], np.pi/4) + min(-orig_obs['knee_l_rot'], np.pi/4)) / 10
            return orig_reward + knee_reward
        elif mode == 'mulknee':
            knee_reward = min(-orig_obs['knee_r_rot'], np.pi/4) + min(-orig_obs['knee_l_rot'], np.pi/4) / 0.5 + 0.5
            if orig_reward > 0:
                reward = orig_reward * max(knee_reward, 0.)
            else:
                reward = orig_reward
            # print(f'{self.t:3d} {reward:+2.4f} {orig_reward:+2.4f} {knee_reward:+2.4f}')
            return reward
        elif mode == 'smallsteps':
            feet_dist = abs(orig_obs['talus_l_x'] - orig_obs['talus_r_x'])
            feet_dist_penalty = -max(feet_dist - 0.6, 0) / 3
            return orig_reward + feet_dist_penalty
        else:
            raise ValueError('Unknown mode `%s`' % mode)


class FilteredWalker(gym.Wrapper):
    """ Wraps the Walker to produce filtered observations """
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, env, extrapolation_order=2):
        super(FilteredWalker, self).__init__(env)
        from turnips.obs_filter import VectorObservationFilter
        self.extrapolation_order = extrapolation_order
        self.filter = VectorObservationFilter(extrapolation_order)
        self.filtered_memory = {name: [] for name in ORIG_NAMES}
        self.t = 0

    def _reset(self):
        obs = self.env.reset()
        if isinstance(obs, bool) and obs == False:
            return obs
        self.t = 0
        self.filter.reset()
        obs = self.filter.step(obs)
        self.update_memory()
        return obs

    def _step(self, action):
        self.t += 1
        obs, reward, done, info = self.env.step(action)
        obs = self.filter.step(obs)
        self.update_memory()
        return obs, reward, done, info

    def update_memory(self):
        # For each obs variable [1,2,3], pad it from left [1,...,1,1,1,2,3], reverse it [3,2,1,1,...1,1]. Current observation is at 0.
        if self.t < 10:
            m = {name: np.flip(np.pad(self.filter.filters[name].extrapolated, mode='edge', pad_width=(32, 0)), axis=0) for name in ORIG_NAMES}
        else:
            m = {name: np.flip(self.filter.filters[name].extrapolated, axis=0) for name in ORIG_NAMES}
        self.filtered_memory = m


class MuscleWalker(gym.ActionWrapper):
    def __init__(self, env):
        super(MuscleWalker, self).__init__(env)
        self.action_space = gym.spaces.Box(np.zeros(10), np.ones(10))

    def _action_leg(self, action):
        def left(a):
            return 1 - a * 2 if a < 0.5 else 0

        def right(a):
            return (a - 0.5) * 2 if a >= 0.5 else 0

        res = [0] * 9
        res[0] = left(action[0])
        res[4] = right(action[0])
        res[1] = left(action[1])
        res[5] = right(action[1])
        res[2] = left(action[2])
        res[3] = right(action[2])
        res[7] = left(action[3])
        res[8] = right(action[3])
        res[6] = action[4]
        return res

    def _action(self, action):
        right = self._action_leg(action[:5])
        left = self._action_leg(action[5:])
        return right + left

    def _reverse_action(self, action):
        assert False
        pass


class RepeatActionsWalker(gym.Wrapper):
    def __init__(self, env, repeat=2):
        """Return only every `skip`-th frame"""
        super(RepeatActionsWalker, self).__init__(env)
        self._repeat = repeat

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = total_origreward = 0.0
        done = None
        for _ in range(self._repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            total_origreward += info['original_reward']
            if done:
                break
        info['original_reward'] = total_origreward
        return obs, total_reward, done, info

def get_random_token():
    s = random.getstate()
    random.seed()
    token = ''.join(random.choice('0123456789abcdef') for n in range(30))
    random.setstate(s)
    return token


class RunEnvWrapper(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, runenv, difficulty):
        super(RunEnvWrapper, self).__init__()
        self.runenv = runenv
        self.difficulty = difficulty
        self.action_space = runenv.action_space
        self.observation_space = runenv.observation_space

        self.next_seed = None
        self._seed()

    def inject_seed(self, seed):
        self.next_seed = seed

    def _reset(self):
        rand_state = np.random.get_state()  # Because the environment uses the global random generator
        if self.next_seed is None:
            obs = self.runenv.reset(difficulty=self.difficulty, seed=self.np_random.randint(0, 2**32))
        else:
            print('LOADING INJECTED SEED', self.next_seed)
            obs = self.runenv.reset(difficulty=self.difficulty, seed=self.next_seed)
            self.next_seed = None

        np.random.set_state(rand_state)
        return obs

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        actual_action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.runenv.step(actual_action)

    def _close(self):
        return self.runenv.close()

class VirtualEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, file_names):
        super(VirtualEnv, self).__init__()
        self.file_names = file_names
        self.action_space = Box(low=0, high=1, shape=[18])
        self.observation_space = Box(low=-3, high=+3, shape=[41])
        self.next_action = None
        self.last_episode = False

    def _reset(self):
        success = False
        fn = None
        while not success:
            try:
                if len(self.file_names) == 0:
                    return None
                fn = self.file_names[0]
                self.file_names = self.file_names[1:]
                if len(self.file_names) == 0:
                    self.last_episode = True

                with h5py.File(fn, "r") as f:
                    self.actions = f['actions'][:]
                    self.observations = f['observations'][:]
                    self.rewards = f['rewards'][:]
                success = True
            except Exception as e:
                print("Cannot read ", fn, str(e))

        self.t = 0
        self.next_action = self.actions[0]
        return self.observations[0]

    def _step(self, action):
        self.t += 1
        done = self.t == len(self.actions)
        self.next_action = self.actions[self.t] if not done else None
        return self.observations[self.t], self.rewards[self.t], done, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class VirtualWalker(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, file_names):
        super(VirtualWalker, self).__init__()
        self.file_names = file_names
        self.action_space = Box(low=0, high=1, shape=[18])
        self.observation_space = Box(low=-3, high=+3, shape=[41])
        self.next_action = None

    def _reset(self):
        success = False
        fn = None
        while not success:
            try:
                fn = self.np_random.choice(self.file_names)
                with h5py.File(fn, "r") as f:
                    self.actions = f['actions'][:]
                    self.observations = f['observations'][:]
                    self.rewards = f['rewards'][:]
                success = True
            except Exception as e:
                print("Cannot read ", fn, str(e))

        self.t = 0
        self.next_action = self.actions[0]
        return self.observations[0]

    def _step(self, action):
        assert np.allclose(action, self.next_action)
        self.t += 1
        done = self.t == len(self.actions)
        self.next_action = self.actions[self.t] if not done else None
        return self.observations[self.t], self.rewards[self.t], done, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class h5pyEnvLogger(gym.Wrapper):
    """ Wraps any environment saving the observations and actions to a dir/[filename_prefix][date-unique-string].h5py """
    def __init__(self, env, log_dir, filename_prefix="", additional_info={}):
        super(self.__class__, self).__init__(env)
        self.log_dir = log_dir
        self.filename_prefix = filename_prefix
        self.additional_info = additional_info

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        self.f = None
        atexit.register(self._try_save)  # If the last one was not properly finished (with done=True)

    def _reset(self):
        obs = self.env.reset()
        if isinstance(obs, bool) and obs == False:
            return obs

        self._try_save(done=False)
        self.f = self._get_new_file()

        self.obs_seq = [obs]
        self.rew_seq = [0]
        self.action_seq = []
        return obs

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.action_seq.append(action)
        self.obs_seq.append(obs)
        self.rew_seq.append(reward)
        # print(len(self.obs_seq))
        if done:
            self._try_save(done=True)
        return obs, reward, done, info

    def _close(self):
        self._try_save(done=False)
        if self.env:
            self.env.close()

    def _get_new_file(self):
        token = get_random_token()
        now = datetime.now()
        fname = self.filename_prefix + "{:%Y-%m-%d-%H-%M-%S}_{}.hdf5".format(now, token)

        f = h5py.File(path.join(self.log_dir, fname))
        f.attrs['timestamp'] = int(time.time())
        f.attrs['host'] = socket.gethostname()
        f.attrs['user'] = os.environ['USER']
        for k, v in self.additional_info.items():
            f.attrs[k] = str(v)
        return f

    def _try_save(self, done=False):
        if self.f is None:
            return
        self.f.attrs['done'] = done
        self.f.create_dataset("observations", data=np.asarray(self.obs_seq, dtype=np.float32), compression="gzip")
        self.f.create_dataset("actions", data=np.asarray(self.action_seq, dtype=np.float32), compression="gzip")
        self.f.create_dataset("rewards", data=np.asarray(self.rew_seq, dtype=np.float32), compression="gzip")
        if hasattr(self.env, 'obs_names'):
            self.f.attrs['obs_names'] = str(self.env.obs_names)
        self.f.close()
        self.f = None

new8_mean = {'strength_l': 1.0097175, 'strength_r': 0.90238577, 'obst_x': 2.444555, 'obst_y': 0.0010591592, 'obst_r': 0.022993524, 'no_obstacles': 0.73976523, 'pelvis1_rot': 0.071717933, 'd1_pelvis1_rot': -0.00012956825, 'd2_pelvis1_rot': -9.5867901e-05, 'd3_pelvis1_rot': -6.1142935e-05, 'd4_pelvis1_rot': -2.6487545e-05, 'd5_pelvis1_rot': 6.4619771e-06, 'd6_pelvis1_rot': 3.6258964e-05, 'd7_pelvis1_rot': 6.1636318e-05, 'hip_r_rot': -0.1299431, 'd1_hip_r_rot': -0.00099760958, 'd2_hip_r_rot': -0.0010296527, 'd3_hip_r_rot': -0.0010596006, 'd4_hip_r_rot': -0.0010828037, 'd5_hip_r_rot': -0.0010939358, 'd6_hip_r_rot': -0.0010926976, 'd7_hip_r_rot': -0.0010772172, 'hip_l_rot': -0.14366169, 'd1_hip_l_rot': 0.00038295158, 'd2_hip_l_rot': 0.00037394505, 'd3_hip_l_rot': 0.00035138673, 'd4_hip_l_rot': 0.0003165268, 'd5_hip_l_rot': 0.0002756397, 'd6_hip_l_rot': 0.00023324385, 'd7_hip_l_rot': 0.00019421059, 'knee_r_rot': -0.77886003, 'd1_knee_r_rot': -3.6942773e-05, 'd2_knee_r_rot': -1.5712898e-05, 'd3_knee_r_rot': 1.3722648e-06, 'd4_knee_r_rot': 3.8169924e-06, 'd5_knee_r_rot': -1.1872157e-05, 'd6_knee_r_rot': -4.3672324e-05, 'd7_knee_r_rot': -9.1686103e-05, 'knee_l_rot': -0.744937, 'd1_knee_l_rot': -0.001425802, 'd2_knee_l_rot': -0.0015150046, 'd3_knee_l_rot': -0.0015982105, 'd4_knee_l_rot': -0.0016633634, 'd5_knee_l_rot': -0.0017104479, 'd6_knee_l_rot': -0.0017371241, 'd7_knee_l_rot': -0.0017635755, 'ankle_r_rot': -0.22953671, 'd1_ankle_r_rot': -0.00099861086, 'd2_ankle_r_rot': -0.0010346439, 'd3_ankle_r_rot': -0.0010545332, 'd4_ankle_r_rot': -0.0010593106, 'd5_ankle_r_rot': -0.0010861743, 'd6_ankle_r_rot': -0.001083052, 'd7_ankle_r_rot': -0.0010846574, 'ankle_l_rot': -0.27226624, 'd1_ankle_l_rot': -0.00025583943, 'd2_ankle_l_rot': -0.00021112135, 'd3_ankle_l_rot': -0.00014415372, 'd4_ankle_l_rot': -0.00010901121, 'd5_ankle_l_rot': -9.8902041e-05, 'd6_ankle_l_rot': -0.00011740458, 'd7_ankle_l_rot': -0.00011617777, 'pelvis_y': 0.88075048, 'd1_pelvis_y': -0.00016484197, 'd2_pelvis_y': -0.000155346, 'd3_pelvis_y': -0.00014588848, 'd4_pelvis_y': -0.00013638132, 'd5_pelvis_y': -0.00012692223, 'd6_pelvis_y': -0.00011793638, 'd7_pelvis_y': -0.0001098477, 'mass_y': 0.9568345, 'd1_mass_y': -0.00010835366, 'd2_mass_y': -0.00010302355, 'd3_mass_y': -9.7935874e-05, 'd4_mass_y': -9.288478e-05, 'd5_mass_y': -8.776586e-05, 'd6_mass_y': -8.2664475e-05, 'd7_mass_y': -7.7759367e-05, 'pelvis1_rot_v': -0.014682037, 'd1_pelvis1_rot_v': -0.0032798268, 'd2_pelvis1_rot_v': -0.0034337144, 'd3_pelvis1_rot_v': -0.0034942233, 'd4_pelvis1_rot_v': -0.0034070646, 'd5_pelvis1_rot_v': -0.0031604238, 'd6_pelvis1_rot_v': -0.0027673135, 'd7_pelvis1_rot_v': -0.0023364781, 'hip_r_rot_v': -0.097990967, 'd1_hip_r_rot_v': 0.0031930995, 'd2_hip_r_rot_v': 0.0031475616, 'd3_hip_r_rot_v': 0.0027783129, 'd4_hip_r_rot_v': 0.0017467759, 'd5_hip_r_rot_v': 0.00050890195, 'd6_hip_r_rot_v': -0.0008313416, 'd7_hip_r_rot_v': -0.002093483, 'hip_l_rot_v': 0.038907681, 'd1_hip_l_rot_v': 0.00019368305, 'd2_hip_l_rot_v': 0.0015064032, 'd3_hip_l_rot_v': 0.0029702382, 'd4_hip_l_rot_v': 0.0038809539, 'd5_hip_l_rot_v': 0.0042196652, 'd6_hip_l_rot_v': 0.0041856356, 'd7_hip_l_rot_v': 0.0037752734, 'knee_r_rot_v': -0.0042337002, 'd1_knee_r_rot_v': -0.0020850194, 'd2_knee_r_rot_v': -0.0020545004, 'd3_knee_r_rot_v': -0.0011903398, 'd4_knee_r_rot_v': 0.00078396162, 'd5_knee_r_rot_v': 0.0023082171, 'd6_knee_r_rot_v': 0.0040824469, 'd7_knee_r_rot_v': 0.005304704, 'knee_l_rot_v': -0.13748577, 'd1_knee_l_rot_v': 0.0088413684, 'd2_knee_l_rot_v': 0.0091763064, 'd3_knee_l_rot_v': 0.0074180807, 'd4_knee_l_rot_v': 0.0056213904, 'd5_knee_l_rot_v': 0.0037690294, 'd6_knee_l_rot_v': 0.0015799658, 'd7_knee_l_rot_v': 0.0035136319, 'ankle_r_rot_v': -0.10288398, 'd1_ankle_r_rot_v': 0.0036526276, 'd2_ankle_r_rot_v': 0.0031935016, 'd3_ankle_r_rot_v': 0.00065479748, 'd4_ankle_r_rot_v': 0.0014198867, 'd5_ankle_r_rot_v': 0.0022759933, 'd6_ankle_r_rot_v': -0.0015950868, 'd7_ankle_r_rot_v': 0.00071777723, 'ankle_l_rot_v': -0.034613069, 'd1_ankle_l_rot_v': -0.0010516803, 'd2_ankle_l_rot_v': -0.0087354882, 'd3_ankle_l_rot_v': -0.005031086, 'd4_ankle_l_rot_v': -0.0021843493, 'd5_ankle_l_rot_v': 0.00040490329, 'd6_ankle_l_rot_v': 0.0029823212, 'd7_ankle_l_rot_v': -0.0036392682, 'pelvis1_y_v': -0.016922619, 'd1_pelvis1_y_v': -0.00095045712, 'd2_pelvis1_y_v': -0.00094598363, 'd3_pelvis1_y_v': -0.00094731164, 'd4_pelvis1_y_v': -0.00095321017, 'd5_pelvis1_y_v': -0.00093039678, 'd6_pelvis1_y_v': -0.00086127332, 'd7_pelvis1_y_v': -0.00074451818, 'mass_y_v': -0.01111523, 'd1_mass_y_v': -0.00055009912, 'd2_mass_y_v': -0.00051764306, 'd3_mass_y_v': -0.00050352042, 'd4_mass_y_v': -0.00050880871, 'd5_mass_y_v': -0.00051380723, 'd6_mass_y_v': -0.0005014076, 'd7_mass_y_v': -0.00046728886, 'mass_x_v': 2.6535866, 'd1_mass_x_v': 0.0042943358, 'd2_mass_x_v': 0.0042973235, 'd3_mass_x_v': 0.0043025403, 'd4_mass_x_v': 0.0043088146, 'd5_mass_x_v': 0.0043143975, 'd6_mass_x_v': 0.004323747, 'd7_mass_x_v': 0.0043586148, 'pelvis1_x_v': 2.6570776, 'd1_pelvis1_x_v': 0.0037045823, 'd2_pelvis1_x_v': 0.0036688759, 'd3_pelvis1_x_v': 0.0036517298, 'd4_pelvis1_x_v': 0.003674255, 'd5_pelvis1_x_v': 0.0037382576, 'd6_pelvis1_x_v': 0.0038314371, 'd7_pelvis1_x_v': 0.0039335378, 'torso_y': 0.95402819, 'd1_torso_y': -0.00015682046, 'd2_torso_y': -0.00015590763, 'd3_torso_y': -0.00014899836, 'd4_torso_y': -0.0001420444, 'd5_torso_y': -0.0001353167, 'd6_torso_y': -0.00012890089, 'd7_torso_y': -0.00012300028, 'head_y': 1.4967523, 'd1_head_y': -0.00022128619, 'd2_head_y': -0.00021712831, 'd3_head_y': -0.000204165, 'd4_head_y': -0.00019241733, 'd5_head_y': -0.00018168849, 'd6_head_y': -0.00017158742, 'd7_head_y': -0.00016240076, 'toe_l_y': 0.077669993, 'd1_toe_l_y': 0.00022288212, 'd2_toe_l_y': 0.00022744077, 'd3_toe_l_y': 0.00021192062, 'd4_toe_l_y': 0.00021033509, 'd5_toe_l_y': 0.00021371017, 'd6_toe_l_y': 0.00021818026, 'd7_toe_l_y': 0.00022875694, 'toe_r_y': 0.084211923, 'd1_toe_r_y': 0.00018040175, 'd2_toe_r_y': 0.00018504562, 'd3_toe_r_y': 0.00016525798, 'd4_toe_r_y': 0.00015771226, 'd5_toe_r_y': 0.00015259207, 'd6_toe_r_y': 0.00014682916, 'd7_toe_r_y': 0.00014109803, 'talus_l_y': 0.20029706, 'd1_talus_l_y': 0.00032076638, 'd2_talus_l_y': 0.00030034609, 'd3_talus_l_y': 0.00029842355, 'd4_talus_l_y': 0.00030585236, 'd5_talus_l_y': 0.00031508942, 'd6_talus_l_y': 0.00032486158, 'd7_talus_l_y': 0.00033925104, 'talus_r_y': 0.2075, 'd1_talus_r_y': 0.000324981, 'd2_talus_r_y': 0.00029992286, 'd3_talus_r_y': 0.00028404428, 'd4_talus_r_y': 0.00027945181, 'd5_talus_r_y': 0.00027432392, 'd6_talus_r_y': 0.00026881698, 'd7_talus_r_y': 0.00026291059, 'rel_head_x': -0.068142749, 'd1_head_x': 0.026621748, 'd2_head_x': 0.02657097, 'd3_head_x': 0.02651329, 'd4_head_x': 0.026455978, 'd5_head_x': 0.026399581, 'd6_head_x': 0.026344338, 'd7_head_x': 0.026290486, 'rel_torso_x': -0.10511455, 'd1_torso_x': 0.026566584, 'd2_torso_x': 0.026531635, 'd3_torso_x': 0.026493033, 'd4_torso_x': 0.026453283, 'd5_torso_x': 0.026413346, 'd6_torso_x': 0.026372902, 'd7_torso_x': 0.026331808, 'rel_toe_l_x': -0.371039, 'd1_toe_l_x': 0.026109898, 'd2_toe_l_x': 0.026051123, 'd3_toe_l_x': 0.025997683, 'd4_toe_l_x': 0.025943257, 'd5_toe_l_x': 0.025890499, 'd6_toe_l_x': 0.025838368, 'd7_toe_l_x': 0.025785195, 'rel_toe_r_x': -0.36996946, 'd1_toe_r_x': 0.025501408, 'd2_toe_r_x': 0.025495348, 'd3_toe_r_x': 0.025462303, 'd4_toe_r_x': 0.025430951, 'd5_toe_r_x': 0.025397187, 'd6_toe_r_x': 0.02536672, 'd7_toe_r_x': 0.025335217, 'rel_talus_l_x': -0.38906729, 'd1_talus_l_x': 0.026243977, 'd2_talus_l_x': 0.026180809, 'd3_talus_l_x': 0.026140885, 'd4_talus_l_x': 0.026096577, 'd5_talus_l_x': 0.026050419, 'd6_talus_l_x': 0.02600465, 'd7_talus_l_x': 0.025955789, 'rel_talus_r_x': -0.39119512, 'd1_talus_r_x': 0.025746204, 'd2_talus_r_x': 0.025721991, 'd3_talus_r_x': 0.025699992, 'd4_talus_r_x': 0.025671501, 'd5_talus_r_x': 0.025641052, 'd6_talus_r_x': 0.025610367, 'd7_talus_r_x': 0.025580097, 'rel_mass_x': -0.11602429, 'd1_mass_x': 0.026514219, 'd2_mass_x': 0.02647127, 'd3_mass_x': 0.026428265, 'd4_mass_x': 0.026385214, 'd5_mass_x': 0.026342103, 'd6_mass_x': 0.02629892, 'd7_mass_x': 0.02625552, 'dist_talus_l_talus_x': 0.0021280271, 'dist_talus_r_talus_x': -0.0021280271, 'dist_talus_l_talus_y': -0.0072027673, 'dist_talus_r_talus_y': 0.0072027673, 'd_dist_talus_l_talus_x': 0.00049773074, 'd_dist_talus_r_talus_x': -0.00049773074, 'd_dist_talus_l_talus_y': -4.2141796e-06, 'd_dist_talus_r_talus_y': 4.2141796e-06, 'dist_toe_r_beg_obst_x': 2.4701617, 'dist_toe_l_beg_obst_x': 2.5023909, 'dist_toe_r_beg_obst_y': 2.2170906, 'dist_toe_l_beg_obst_y': 2.2095437, 'dist_talus_r_end_obst_x': 2.480113, 'dist_talus_l_end_obst_x': 2.5022948, 'dist_talus_r_end_obst_y': 2.1538005, 'dist_talus_l_end_obst_y': 2.1476228, 'muscle_activation_0_r_0': 0.86128038, 'muscle_activation_0_r_1': 0.58600634, 'muscle_activation_0_r_2': 0.56207192, 'muscle_activation_0_r_3': 0.55006105, 'muscle_activation_0_r_4': 0.26114738, 'muscle_activation_0_r_5': 0.47988868, 'muscle_activation_0_r_6': 0.43132609, 'muscle_activation_0_r_7': 0.43585414, 'muscle_activation_0_r_8': 0.51006174, 'muscle_activation_0_l_0': 0.84089601, 'muscle_activation_0_l_1': 0.56877929, 'muscle_activation_0_l_2': 0.62392545, 'muscle_activation_0_l_3': 0.56340516, 'muscle_activation_0_l_4': 0.2631509, 'muscle_activation_0_l_5': 0.50207359, 'muscle_activation_0_l_6': 0.42160198, 'muscle_activation_0_l_7': 0.47221753, 'muscle_activation_0_l_8': 0.47623181, 'muscle_activation_1_r_0': 0.86025703, 'muscle_activation_1_r_1': 0.58533788, 'muscle_activation_1_r_2': 0.56096828, 'muscle_activation_1_r_3': 0.54929996, 'muscle_activation_1_r_4': 0.260351, 'muscle_activation_1_r_5': 0.47923306, 'muscle_activation_1_r_6': 0.43120024, 'muscle_activation_1_r_7': 0.43527141, 'muscle_activation_1_r_8': 0.50941682, 'muscle_activation_1_l_0': 0.84001815, 'muscle_activation_1_l_1': 0.56763393, 'muscle_activation_1_l_2': 0.62297654, 'muscle_activation_1_l_3': 0.5630601, 'muscle_activation_1_l_4': 0.26293564, 'muscle_activation_1_l_5': 0.50100404, 'muscle_activation_1_l_6': 0.42107257, 'muscle_activation_1_l_7': 0.47115728, 'muscle_activation_1_l_8': 0.47553584, 'muscle_activation_2_r_0': 0.85925543, 'muscle_activation_2_r_1': 0.58463454, 'muscle_activation_2_r_2': 0.55982053, 'muscle_activation_2_r_3': 0.54846656, 'muscle_activation_2_r_4': 0.25957695, 'muscle_activation_2_r_5': 0.47854421, 'muscle_activation_2_r_6': 0.43102756, 'muscle_activation_2_r_7': 0.43461064, 'muscle_activation_2_r_8': 0.50886136, 'muscle_activation_2_l_0': 0.83912909, 'muscle_activation_2_l_1': 0.56651217, 'muscle_activation_2_l_2': 0.62205243, 'muscle_activation_2_l_3': 0.56269413, 'muscle_activation_2_l_4': 0.26286748, 'muscle_activation_2_l_5': 0.50010318, 'muscle_activation_2_l_6': 0.42049265, 'muscle_activation_2_l_7': 0.47008413, 'muscle_activation_2_l_8': 0.47488585, 'muscle_activation_3_r_0': 0.85834682, 'muscle_activation_3_r_1': 0.58387655, 'muscle_activation_3_r_2': 0.55859429, 'muscle_activation_3_r_3': 0.54751247, 'muscle_activation_3_r_4': 0.25873074, 'muscle_activation_3_r_5': 0.47780594, 'muscle_activation_3_r_6': 0.4307887, 'muscle_activation_3_r_7': 0.43391347, 'muscle_activation_3_r_8': 0.50846964, 'muscle_activation_3_l_0': 0.83822495, 'muscle_activation_3_l_1': 0.56533313, 'muscle_activation_3_l_2': 0.62117827, 'muscle_activation_3_l_3': 0.56230325, 'muscle_activation_3_l_4': 0.26275015, 'muscle_activation_3_l_5': 0.49912187, 'muscle_activation_3_l_6': 0.41991729, 'muscle_activation_3_l_7': 0.46899354, 'muscle_activation_3_l_8': 0.47420794, 'muscle_activation_4_r_0': 0.85735804, 'muscle_activation_4_r_1': 0.58313918, 'muscle_activation_4_r_2': 0.55731875, 'muscle_activation_4_r_3': 0.54668212, 'muscle_activation_4_r_4': 0.25776738, 'muscle_activation_4_r_5': 0.47698483, 'muscle_activation_4_r_6': 0.43057877, 'muscle_activation_4_r_7': 0.43316147, 'muscle_activation_4_r_8': 0.50804603, 'muscle_activation_4_l_0': 0.83729827, 'muscle_activation_4_l_1': 0.56427938, 'muscle_activation_4_l_2': 0.62044609, 'muscle_activation_4_l_3': 0.56188989, 'muscle_activation_4_l_4': 0.26264721, 'muscle_activation_4_l_5': 0.49809903, 'muscle_activation_4_l_6': 0.41940838, 'muscle_activation_4_l_7': 0.46791139, 'muscle_activation_4_l_8': 0.47349116, 'muscle_activation_5_r_0': 0.85656667, 'muscle_activation_5_r_1': 0.58233792, 'muscle_activation_5_r_2': 0.55604857, 'muscle_activation_5_r_3': 0.54580879, 'muscle_activation_5_r_4': 0.25669727, 'muscle_activation_5_r_5': 0.4761171, 'muscle_activation_5_r_6': 0.4302882, 'muscle_activation_5_r_7': 0.43231609, 'muscle_activation_5_r_8': 0.50756216, 'muscle_activation_5_l_0': 0.8363353, 'muscle_activation_5_l_1': 0.56319714, 'muscle_activation_5_l_2': 0.61970329, 'muscle_activation_5_l_3': 0.56143069, 'muscle_activation_5_l_4': 0.26247656, 'muscle_activation_5_l_5': 0.49711996, 'muscle_activation_5_l_6': 0.4190495, 'muscle_activation_5_l_7': 0.46686915, 'muscle_activation_5_l_8': 0.47271717, 'muscle_activation_6_r_0': 0.8559413, 'muscle_activation_6_r_1': 0.58142996, 'muscle_activation_6_r_2': 0.55478835, 'muscle_activation_6_r_3': 0.54486454, 'muscle_activation_6_r_4': 0.25553727, 'muscle_activation_6_r_5': 0.4751682, 'muscle_activation_6_r_6': 0.43000042, 'muscle_activation_6_r_7': 0.43139896, 'muscle_activation_6_r_8': 0.50721782, 'muscle_activation_6_l_0': 0.83540946, 'muscle_activation_6_l_1': 0.56215864, 'muscle_activation_6_l_2': 0.61908001, 'muscle_activation_6_l_3': 0.56090564, 'muscle_activation_6_l_4': 0.26239067, 'muscle_activation_6_l_5': 0.49623412, 'muscle_activation_6_l_6': 0.41861537, 'muscle_activation_6_l_7': 0.46594191, 'muscle_activation_6_l_8': 0.47195867, 'muscle_activation_7_r_0': 0.85535455, 'muscle_activation_7_r_1': 0.58056617, 'muscle_activation_7_r_2': 0.55354762, 'muscle_activation_7_r_3': 0.54389417, 'muscle_activation_7_r_4': 0.25430357, 'muscle_activation_7_r_5': 0.47416705, 'muscle_activation_7_r_6': 0.42958543, 'muscle_activation_7_r_7': 0.43045151, 'muscle_activation_7_r_8': 0.50690782, 'muscle_activation_7_l_0': 0.83445203, 'muscle_activation_7_l_1': 0.56124562, 'muscle_activation_7_l_2': 0.61851603, 'muscle_activation_7_l_3': 0.56041265, 'muscle_activation_7_l_4': 0.26225221, 'muscle_activation_7_l_5': 0.49539849, 'muscle_activation_7_l_6': 0.41813636, 'muscle_activation_7_l_7': 0.46505874, 'muscle_activation_7_l_8': 0.4711296}
new8_std = {'strength_l': 0.083035946, 'strength_r': 0.1112333, 'obst_x': 0.99352223, 'obst_y': 0.0069834418, 'obst_r': 0.043235678, 'no_obstacles': 0.43877432, 'pelvis1_rot': 0.14851426, 'd1_pelvis1_rot': 0.020511271, 'd2_pelvis1_rot': 0.020485021, 'd3_pelvis1_rot': 0.020457236, 'd4_pelvis1_rot': 0.020428732, 'd5_pelvis1_rot': 0.020401256, 'd6_pelvis1_rot': 0.020375848, 'd7_pelvis1_rot': 0.020354271, 'hip_r_rot': 0.63582432, 'd1_hip_r_rot': 0.067647099, 'd2_hip_r_rot': 0.067634732, 'd3_hip_r_rot': 0.0676236, 'd4_hip_r_rot': 0.067615107, 'd5_hip_r_rot': 0.067608677, 'd6_hip_r_rot': 0.067601182, 'd7_hip_r_rot': 0.067587771, 'hip_l_rot': 0.64111853, 'd1_hip_l_rot': 0.067147039, 'd2_hip_l_rot': 0.0670707, 'd3_hip_l_rot': 0.066984423, 'd4_hip_l_rot': 0.066889226, 'd5_hip_l_rot': 0.066779807, 'd6_hip_l_rot': 0.066654742, 'd7_hip_l_rot': 0.066521659, 'knee_r_rot': 0.70040089, 'd1_knee_r_rot': 0.07001926, 'd2_knee_r_rot': 0.070001312, 'd3_knee_r_rot': 0.06998466, 'd4_knee_r_rot': 0.06997139, 'd5_knee_r_rot': 0.06995137, 'd6_knee_r_rot': 0.069921136, 'd7_knee_r_rot': 0.069866695, 'knee_l_rot': 0.68159336, 'd1_knee_l_rot': 0.070324555, 'd2_knee_l_rot': 0.070218526, 'd3_knee_l_rot': 0.070111752, 'd4_knee_l_rot': 0.070032038, 'd5_knee_l_rot': 0.069969095, 'd6_knee_l_rot': 0.069911785, 'd7_knee_l_rot': 0.069824338, 'ankle_r_rot': 0.42797402, 'd1_ankle_r_rot': 0.051476967, 'd2_ankle_r_rot': 0.051454145, 'd3_ankle_r_rot': 0.051443331, 'd4_ankle_r_rot': 0.051442891, 'd5_ankle_r_rot': 0.051418919, 'd6_ankle_r_rot': 0.051416863, 'd7_ankle_r_rot': 0.051412094, 'ankle_l_rot': 0.42980844, 'd1_ankle_l_rot': 0.051731754, 'd2_ankle_l_rot': 0.051697675, 'd3_ankle_l_rot': 0.051594079, 'd4_ankle_l_rot': 0.051546652, 'd5_ankle_l_rot': 0.051509731, 'd6_ankle_l_rot': 0.051458981, 'd7_ankle_l_rot': 0.051360205, 'pelvis_y': 0.043499999, 'd1_pelvis_y': 0.0058138035, 'd2_pelvis_y': 0.0058041862, 'd3_pelvis_y': 0.0057958937, 'd4_pelvis_y': 0.005788309, 'd5_pelvis_y': 0.0057809539, 'd6_pelvis_y': 0.0057738195, 'd7_pelvis_y': 0.0057669245, 'mass_y': 0.039808668, 'd1_mass_y': 0.0050886408, 'd2_mass_y': 0.0050825789, 'd3_mass_y': 0.0050774445, 'd4_mass_y': 0.0050726589, 'd5_mass_y': 0.0050676237, 'd6_mass_y': 0.0050620269, 'd7_mass_y': 0.0050559579, 'pelvis1_rot_v': 2.0560231, 'd1_pelvis1_rot_v': 0.43148533, 'd2_pelvis1_rot_v': 0.43132386, 'd3_pelvis1_rot_v': 0.43119571, 'd4_pelvis1_rot_v': 0.43107194, 'd5_pelvis1_rot_v': 0.43087292, 'd6_pelvis1_rot_v': 0.43048179, 'd7_pelvis1_rot_v': 0.4300749, 'hip_r_rot_v': 6.7778177, 'd1_hip_r_rot_v': 1.4067084, 'd2_hip_r_rot_v': 1.4066342, 'd3_hip_r_rot_v': 1.4065322, 'd4_hip_r_rot_v': 1.4059629, 'd5_hip_r_rot_v': 1.4051137, 'd6_hip_r_rot_v': 1.4039867, 'd7_hip_r_rot_v': 1.4028198, 'hip_l_rot_v': 6.7304411, 'd1_hip_l_rot_v': 1.4306949, 'd2_hip_l_rot_v': 1.4295769, 'd3_hip_l_rot_v': 1.4282715, 'd4_hip_l_rot_v': 1.4266508, 'd5_hip_l_rot_v': 1.4241227, 'd6_hip_l_rot_v': 1.4208595, 'd7_hip_l_rot_v': 1.4202225, 'knee_r_rot_v': 7.025466, 'd1_knee_r_rot_v': 1.7667654, 'd2_knee_r_rot_v': 1.766711, 'd3_knee_r_rot_v': 1.7663786, 'd4_knee_r_rot_v': 1.764594, 'd5_knee_r_rot_v': 1.7630837, 'd6_knee_r_rot_v': 1.7614598, 'd7_knee_r_rot_v': 1.7589711, 'knee_l_rot_v': 7.0589542, 'd1_knee_l_rot_v': 1.8382046, 'd2_knee_l_rot_v': 1.8367753, 'd3_knee_l_rot_v': 1.8351817, 'd4_knee_l_rot_v': 1.8327742, 'd5_knee_l_rot_v': 1.8284984, 'd6_knee_l_rot_v': 1.8245802, 'd7_knee_l_rot_v': 1.8107365, 'ankle_r_rot_v': 5.3341312, 'd1_ankle_r_rot_v': 4.0947399, 'd2_ankle_r_rot_v': 4.0940609, 'd3_ankle_r_rot_v': 4.0925174, 'd4_ankle_r_rot_v': 4.0922446, 'd5_ankle_r_rot_v': 4.0902076, 'd6_ankle_r_rot_v': 4.0796876, 'd7_ankle_r_rot_v': 4.0673175, 'ankle_l_rot_v': 5.5296516, 'd1_ankle_l_rot_v': 4.4483781, 'd2_ankle_l_rot_v': 4.424262, 'd3_ankle_l_rot_v': 4.4208841, 'd4_ankle_l_rot_v': 4.4195933, 'd5_ankle_l_rot_v': 4.4187131, 'd6_ankle_l_rot_v': 4.4161601, 'd7_ankle_l_rot_v': 4.3997593, 'pelvis1_y_v': 0.58278751, 'd1_pelvis1_y_v': 0.11329156, 'd2_pelvis1_y_v': 0.11319657, 'd3_pelvis1_y_v': 0.11308303, 'd4_pelvis1_y_v': 0.1129785, 'd5_pelvis1_y_v': 0.11290009, 'd6_pelvis1_y_v': 0.11282613, 'd7_pelvis1_y_v': 0.1127511, 'mass_y_v': 0.50998801, 'd1_mass_y_v': 0.097186483, 'd2_mass_y_v': 0.097089775, 'd3_mass_y_v': 0.096979916, 'd4_mass_y_v': 0.096858568, 'd5_mass_y_v': 0.096757889, 'd6_mass_y_v': 0.096704043, 'd7_mass_y_v': 0.096676983, 'mass_x_v': 0.72253585, 'd1_mass_x_v': 0.028709223, 'd2_mass_x_v': 0.028702952, 'd3_mass_x_v': 0.028693918, 'd4_mass_x_v': 0.028681319, 'd5_mass_x_v': 0.028667253, 'd6_mass_x_v': 0.028645968, 'd7_mass_x_v': 0.028597817, 'pelvis1_x_v': 0.88618195, 'd1_pelvis1_x_v': 0.13787039, 'd2_pelvis1_x_v': 0.13781941, 'd3_pelvis1_x_v': 0.13778915, 'd4_pelvis1_x_v': 0.13775414, 'd5_pelvis1_x_v': 0.13768013, 'd6_pelvis1_x_v': 0.13754159, 'd7_pelvis1_x_v': 0.13742587, 'torso_y': 0.036774661, 'd1_torso_y': 0.0044543184, 'd2_torso_y': 0.0044340673, 'd3_torso_y': 0.0044277087, 'd4_torso_y': 0.0044207452, 'd5_torso_y': 0.0044143018, 'd6_torso_y': 0.0044080708, 'd7_torso_y': 0.0044019497, 'head_y': 0.043757137, 'd1_head_y': 0.0049440311, 'd2_head_y': 0.0048997067, 'd3_head_y': 0.0048739617, 'd4_head_y': 0.0048497068, 'd5_head_y': 0.0048280666, 'd6_head_y': 0.0048085442, 'd7_head_y': 0.0047915736, 'toe_l_y': 0.094213568, 'd1_toe_l_y': 0.012178023, 'd2_toe_l_y': 0.012001485, 'd3_toe_l_y': 0.012047529, 'd4_toe_l_y': 0.012069438, 'd5_toe_l_y': 0.0120691, 'd6_toe_l_y': 0.012062145, 'd7_toe_l_y': 0.012039662, 'toe_r_y': 0.099506974, 'd1_toe_r_y': 0.012875916, 'd2_toe_r_y': 0.012703011, 'd3_toe_r_y': 0.012745638, 'd4_toe_r_y': 0.01275496, 'd5_toe_r_y': 0.012753404, 'd6_toe_r_y': 0.012750514, 'd7_toe_r_y': 0.012748906, 'talus_l_y': 0.09730681, 'd1_talus_l_y': 0.012808456, 'd2_talus_l_y': 0.012708498, 'd3_talus_l_y': 0.012734254, 'd4_talus_l_y': 0.012737047, 'd5_talus_l_y': 0.012730568, 'd6_talus_l_y': 0.012718741, 'd7_talus_l_y': 0.012692977, 'talus_r_y': 0.10054033, 'd1_talus_r_y': 0.013133117, 'd2_talus_r_y': 0.013044855, 'd3_talus_r_y': 0.01306881, 'd4_talus_r_y': 0.013070047, 'd5_talus_r_y': 0.013067381, 'd6_talus_r_y': 0.013064519, 'd7_talus_r_y': 0.013062623, 'rel_head_x': 0.090989053, 'd1_head_x': 0.010348048, 'd2_head_x': 0.010373724, 'd3_head_x': 0.01040997, 'd4_head_x': 0.01044485, 'd5_head_x': 0.010479193, 'd6_head_x': 0.010513591, 'd7_head_x': 0.010548373, 'rel_torso_x': 0.010788548, 'd1_torso_x': 0.0081358021, 'd2_torso_x': 0.0081898719, 'd3_torso_x': 0.0082483143, 'd4_torso_x': 0.0083037941, 'd5_torso_x': 0.0083588883, 'd6_torso_x': 0.0084137591, 'd7_torso_x': 0.0084686428, 'rel_toe_l_x': 0.29161564, 'd1_toe_l_x': 0.028635122, 'd2_toe_l_x': 0.028526591, 'd3_toe_l_x': 0.028520845, 'd4_toe_l_x': 0.028501732, 'd5_toe_l_x': 0.028475774, 'd6_toe_l_x': 0.028449193, 'd7_toe_l_x': 0.028425172, 'rel_toe_r_x': 0.28979528, 'd1_toe_r_x': 0.027852677, 'd2_toe_r_x': 0.027756352, 'd3_toe_r_x': 0.027772496, 'd4_toe_r_x': 0.027786465, 'd5_toe_r_x': 0.027799616, 'd6_toe_r_x': 0.027813477, 'd7_toe_r_x': 0.02782806, 'rel_talus_l_x': 0.23955794, 'd1_talus_l_x': 0.023737365, 'd2_talus_l_x': 0.023711618, 'd3_talus_l_x': 0.02370633, 'd4_talus_l_x': 0.023694847, 'd5_talus_l_x': 0.02367931, 'd6_talus_l_x': 0.023662629, 'd7_talus_l_x': 0.023646802, 'rel_talus_r_x': 0.23943043, 'd1_talus_r_x': 0.023226716, 'd2_talus_r_x': 0.023220044, 'd3_talus_r_x': 0.02323598, 'd4_talus_r_x': 0.023251783, 'd5_talus_r_x': 0.02326826, 'd6_talus_r_x': 0.023284769, 'd7_talus_r_x': 0.02330146, 'rel_mass_x': 0.033728406, 'd1_mass_x': 0.0072582918, 'd2_mass_x': 0.0073243328, 'd3_mass_x': 0.0073893685, 'd4_mass_x': 0.007453531, 'd5_mass_x': 0.00751686, 'd6_mass_x': 0.007579437, 'd7_mass_x': 0.007641301, 'dist_talus_l_talus_x': 0.43402296, 'dist_talus_r_talus_x': 0.43402296, 'dist_talus_l_talus_y': 0.16262315, 'dist_talus_r_talus_y': 0.16262315, 'd_dist_talus_l_talus_x': 0.037723646, 'd_dist_talus_r_talus_x': 0.037723646, 'd_dist_talus_l_talus_y': 0.017783158, 'd_dist_talus_r_talus_y': 0.017783158, 'dist_toe_r_beg_obst_x': 0.95337451, 'dist_toe_l_beg_obst_x': 0.91001493, 'dist_toe_r_beg_obst_y': 1.3148865, 'dist_toe_l_beg_obst_y': 1.3195204, 'dist_talus_r_end_obst_x': 0.92674404, 'dist_talus_l_end_obst_x': 0.90369737, 'dist_talus_r_end_obst_y': 1.3844641, 'dist_talus_l_end_obst_y': 1.3880259, 'muscle_activation_0_r_0': 0.27592513, 'muscle_activation_0_r_1': 0.44769454, 'muscle_activation_0_r_2': 0.42722738, 'muscle_activation_0_r_3': 0.4027538, 'muscle_activation_0_r_4': 0.33180851, 'muscle_activation_0_r_5': 0.4171851, 'muscle_activation_0_r_6': 0.33900765, 'muscle_activation_0_r_7': 0.41880873, 'muscle_activation_0_r_8': 0.44384715, 'muscle_activation_0_l_0': 0.29486835, 'muscle_activation_0_l_1': 0.45069635, 'muscle_activation_0_l_2': 0.41869435, 'muscle_activation_0_l_3': 0.40059027, 'muscle_activation_0_l_4': 0.33759478, 'muscle_activation_0_l_5': 0.42334652, 'muscle_activation_0_l_6': 0.34094656, 'muscle_activation_0_l_7': 0.42427975, 'muscle_activation_0_l_8': 0.44861928, 'muscle_activation_1_r_0': 0.27736419, 'muscle_activation_1_r_1': 0.44788653, 'muscle_activation_1_r_2': 0.4273932, 'muscle_activation_1_r_3': 0.40296465, 'muscle_activation_1_r_4': 0.33138511, 'muscle_activation_1_r_5': 0.41718557, 'muscle_activation_1_r_6': 0.33911422, 'muscle_activation_1_r_7': 0.41877767, 'muscle_activation_1_r_8': 0.44392905, 'muscle_activation_1_l_0': 0.29583073, 'muscle_activation_1_l_1': 0.45086646, 'muscle_activation_1_l_2': 0.41893563, 'muscle_activation_1_l_3': 0.40070528, 'muscle_activation_1_l_4': 0.33754435, 'muscle_activation_1_l_5': 0.42347422, 'muscle_activation_1_l_6': 0.34109268, 'muscle_activation_1_l_7': 0.42427775, 'muscle_activation_1_l_8': 0.44860679, 'muscle_activation_2_r_0': 0.2788308, 'muscle_activation_2_r_1': 0.44809508, 'muscle_activation_2_r_2': 0.42756322, 'muscle_activation_2_r_3': 0.40320015, 'muscle_activation_2_r_4': 0.33098289, 'muscle_activation_2_r_5': 0.41719797, 'muscle_activation_2_r_6': 0.33924517, 'muscle_activation_2_r_7': 0.41870716, 'muscle_activation_2_r_8': 0.44401944, 'muscle_activation_2_l_0': 0.29680827, 'muscle_activation_2_l_1': 0.45104566, 'muscle_activation_2_l_2': 0.41919053, 'muscle_activation_2_l_3': 0.40085194, 'muscle_activation_2_l_4': 0.33752692, 'muscle_activation_2_l_5': 0.42358223, 'muscle_activation_2_l_6': 0.34119529, 'muscle_activation_2_l_7': 0.42427748, 'muscle_activation_2_l_8': 0.44860265, 'muscle_activation_3_r_0': 0.2801756, 'muscle_activation_3_r_1': 0.44831303, 'muscle_activation_3_r_2': 0.42770126, 'muscle_activation_3_r_3': 0.40341786, 'muscle_activation_3_r_4': 0.3305603, 'muscle_activation_3_r_5': 0.41721696, 'muscle_activation_3_r_6': 0.3393926, 'muscle_activation_3_r_7': 0.41864643, 'muscle_activation_3_r_8': 0.4441146, 'muscle_activation_3_l_0': 0.2978133, 'muscle_activation_3_l_1': 0.4512296, 'muscle_activation_3_l_2': 0.41946027, 'muscle_activation_3_l_3': 0.40102723, 'muscle_activation_3_l_4': 0.33746403, 'muscle_activation_3_l_5': 0.42367557, 'muscle_activation_3_l_6': 0.34123945, 'muscle_activation_3_l_7': 0.42427114, 'muscle_activation_3_l_8': 0.44860747, 'muscle_activation_4_r_0': 0.28160354, 'muscle_activation_4_r_1': 0.44853884, 'muscle_activation_4_r_2': 0.42780745, 'muscle_activation_4_r_3': 0.40360314, 'muscle_activation_4_r_4': 0.33003411, 'muscle_activation_4_r_5': 0.41720083, 'muscle_activation_4_r_6': 0.33954498, 'muscle_activation_4_r_7': 0.41858116, 'muscle_activation_4_r_8': 0.44421855, 'muscle_activation_4_l_0': 0.29883406, 'muscle_activation_4_l_1': 0.45137092, 'muscle_activation_4_l_2': 0.41967085, 'muscle_activation_4_l_3': 0.40118268, 'muscle_activation_4_l_4': 0.33743358, 'muscle_activation_4_l_5': 0.42375612, 'muscle_activation_4_l_6': 0.34133288, 'muscle_activation_4_l_7': 0.42416066, 'muscle_activation_4_l_8': 0.44863695, 'muscle_activation_5_r_0': 0.28279394, 'muscle_activation_5_r_1': 0.44879299, 'muscle_activation_5_r_2': 0.42791477, 'muscle_activation_5_r_3': 0.40378851, 'muscle_activation_5_r_4': 0.32939637, 'muscle_activation_5_r_5': 0.41718689, 'muscle_activation_5_r_6': 0.33971214, 'muscle_activation_5_r_7': 0.41846052, 'muscle_activation_5_r_8': 0.44432473, 'muscle_activation_5_l_0': 0.29989228, 'muscle_activation_5_l_1': 0.45150691, 'muscle_activation_5_l_2': 0.41990042, 'muscle_activation_5_l_3': 0.40137342, 'muscle_activation_5_l_4': 0.33733186, 'muscle_activation_5_l_5': 0.42381454, 'muscle_activation_5_l_6': 0.34146184, 'muscle_activation_5_l_7': 0.42407331, 'muscle_activation_5_l_8': 0.44866082, 'muscle_activation_6_r_0': 0.28373399, 'muscle_activation_6_r_1': 0.4490304, 'muscle_activation_6_r_2': 0.4280273, 'muscle_activation_6_r_3': 0.40396544, 'muscle_activation_6_r_4': 0.32864261, 'muscle_activation_6_r_5': 0.41713202, 'muscle_activation_6_r_6': 0.33987531, 'muscle_activation_6_r_7': 0.41829336, 'muscle_activation_6_r_8': 0.44441977, 'muscle_activation_6_l_0': 0.30091617, 'muscle_activation_6_l_1': 0.4516556, 'muscle_activation_6_l_2': 0.42007944, 'muscle_activation_6_l_3': 0.40159518, 'muscle_activation_6_l_4': 0.33732241, 'muscle_activation_6_l_5': 0.42385656, 'muscle_activation_6_l_6': 0.34154516, 'muscle_activation_6_l_7': 0.42397702, 'muscle_activation_6_l_8': 0.44870821, 'muscle_activation_7_r_0': 0.28461358, 'muscle_activation_7_r_1': 0.4492102, 'muscle_activation_7_r_2': 0.42815167, 'muscle_activation_7_r_3': 0.40416685, 'muscle_activation_7_r_4': 0.32779872, 'muscle_activation_7_r_5': 0.41703498, 'muscle_activation_7_r_6': 0.33999628, 'muscle_activation_7_r_7': 0.41811997, 'muscle_activation_7_r_8': 0.44449082, 'muscle_activation_7_l_0': 0.30196795, 'muscle_activation_7_l_1': 0.45178089, 'muscle_activation_7_l_2': 0.42027819, 'muscle_activation_7_l_3': 0.40179783, 'muscle_activation_7_l_4': 0.33727852, 'muscle_activation_7_l_5': 0.42384997, 'muscle_activation_7_l_6': 0.34163967, 'muscle_activation_7_l_7': 0.4239108, 'muscle_activation_7_l_8': 0.44876689}


class New8ObservationFilter(gym.Wrapper):
    # Z-Filters the observations with a constant values (above)
    def __init__(self, env, obs_names):
        super(self.__class__, self).__init__(env)
        self.obs_names = obs_names

    def _filter(self, obs):
        assert len(obs) == len(self.obs_names)
        for i, obs_name in enumerate(self.obs_names):
            if obs_name in new8_mean:
                obs[i] = (obs[i] - new8_mean[obs_name]) / new8_std[obs_name]
        return obs

    def _reset(self):
        obs = self.env.reset()
        if isinstance(obs, bool) and obs == False:
            return obs
        return self._filter(obs)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._filter(obs), reward, done, info

