import gym
from gym import Wrapper
import h5py
import shutil
import os
from os import path
import time
from datetime import datetime
import random
import numpy as np
import atexit
import socket


class MyRunEnvLogger(Wrapper):
    """ Wraps the osim.env.RunEnv environment and saves all the trajectories to disk. One file per episode. """
    def __init__(self, runner_env, log_dir, additional_info=None):
        """ runner_env: osim.env.RunEnv """
        super(MyRunEnvLogger, self).__init__(runner_env)
        self.log_dir = log_dir
        self.additional_info = additional_info if additional_info is not None else {}

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)

        atexit.register(self._save)  # If the last one was not properly finished (with done=True)
        self.saved = True
        self.f = None

    def _reset(self):
        raise ValueError("This should not be called")

    # noinspection PyMethodOverriding
    def reset(self, difficulty, seed):
        obs = self.env.reset(difficulty, seed)
        if isinstance(obs, bool) and obs == False:
            return obs

        # If the last one was not properly finished (with done=True)
        if self.log_dir is not None:
            try:
                if not self.saved:
                    self._save(done=False)

                now = datetime.now()
                s = random.getstate()
                random.seed()
                token = ''.join(random.choice('0123456789abcdef') for n in range(30))
                random.setstate(s)
                fname = "{:%Y-%m-%d-%H-%M-%S}_{:d}_{:03d}_{}.hdf5".format(now, difficulty, seed, token)

                self.filepath = path.join(self.log_dir, fname)
                self.f = h5py.File(self.filepath, "w")
                self.f.attrs['timestamp'] = int(time.time())
                self.f.attrs['difficulty'] = difficulty
                self.f.attrs['seed'] = seed
                self.f.attrs['host'] = socket.gethostname()
                self.f.attrs['user'] = os.environ['USER']
                for k, v in self.additional_info.items():
                    self.f.attrs[k] = str(v)
            except Exception as e:
                print('Sth wrong happened with logging model data: ', str(e))

        self.obs_seq = [obs]
        self.rew_seq = [0]
        self.action_seq = []
        # self.muscle_activations_seq = [self.env.current_muscle_activations]
        # self.delta_x_rew_seq = [0]
        # self.ligament_rew_seq = [0]
        self.saved = False

        return obs

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.action_seq.append(action)
        self.obs_seq.append(obs)
        self.rew_seq.append(reward)
        # self.ligament_rew_seq.append(info['ligament_reward'])
        # self.delta_x_rew_seq.append(info['delta_x_reward'])
        # self.muscle_activations_seq.append(self.env.current_muscle_activations)
        if done:
            self._save(done=True)
        return obs, reward, done, info

    def _close(self):
        super(MyRunEnvLogger, self)._close()

    def _save(self, done=False):
        if self.log_dir is None:
            return
        try:
            if self.f is None:
                return
            self.f.attrs['done'] = done
            self.f.create_dataset("observations", data=np.asarray(self.obs_seq, dtype=np.float32), compression="gzip")
            self.f.create_dataset("actions", data=np.asarray(self.action_seq, dtype=np.float32), compression="gzip")
            self.f.create_dataset("rewards", data=np.asarray(self.rew_seq, dtype=np.float32), compression="gzip")
            # self.f.create_dataset("muscle_activations", data=np.asarray(self.muscle_activations_seq, dtype=np.float32), compression="gzip")
            # self.f.create_dataset("delta_x_rewards", data=np.asarray(self.delta_x_rew_seq, dtype=np.float32), compression="gzip")
            # self.f.create_dataset("ligament_rewards", data=np.asarray(self.ligament_rew_seq, dtype=np.float32), compression="gzip")
            self.f.close()
            self.saved = True
        except Exception as e:
            print('Sth wrong happened with logging model data: ', str(e))
        self.saved = True
