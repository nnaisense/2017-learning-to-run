import gym
import numpy as np
import tensorflow as tf

import baselines.common.tf_util as U
from baselines.common.distributions import make_pdtype, MultiCategoricalPdType, BetaPdType, BernoulliPdType
from baselines.common.mpi_running_mean_std import RunningMeanStd
from turnips.walker import ORIG_SYMMETRIC_IDS


def gather_2d(tensor_2d, col_indices):
    """ return: tensor_2d[:, col_indices]"""
    flat = tf.reshape(tensor_2d, [-1])
    nrows = tf.shape(tensor_2d)[0]
    ncols = tf.shape(tensor_2d)[1]
    add = tf.range(nrows) * ncols
    idx = col_indices + add
    return tf.gather(flat, idx)


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, bound_by_sigmoid=False,
              sigmoid_coef=1., activation='tanh', normalize_obs=True, actions='gaussian',
              avg_norm_symmetry=False, symmetric_interpretation=False, stdclip=5.0, gaussian_bias=False,
              gaussian_from_binary=False, parallel_value=False, pv_layers=2, pv_hid_size=512,
              three=False):
        assert isinstance(ob_space, gym.spaces.Box)

        if actions == 'binary':
            self.pdtype = pdtype = MultiCategoricalPdType(low=np.zeros_like(ac_space.low, dtype=np.int32),
                                                          high=np.ones_like(ac_space.high, dtype=np.int32))
        elif actions == 'beta':
            self.pdtype = pdtype = BetaPdType(low=np.zeros_like(ac_space.low, dtype=np.int32),
                                              high=np.ones_like(ac_space.high, dtype=np.int32))
        elif actions == 'bernoulli':
            self.pdtype = pdtype = BernoulliPdType(ac_space.low.size)
        elif actions == 'gaussian':
            self.pdtype = pdtype = make_pdtype(ac_space)
        elif actions == 'cat_3':
            self.pdtype = pdtype = MultiCategoricalPdType(low=np.zeros_like(ac_space.low, dtype=np.int32),
                                                          high=np.ones_like(ac_space.high, dtype=np.int32) * 2)
        elif actions == 'cat_5':
            self.pdtype = pdtype = MultiCategoricalPdType(low=np.zeros_like(ac_space.low, dtype=np.int32),
                                                          high=np.ones_like(ac_space.high, dtype=np.int32) * 4)
        else:
            assert False

        sequence_length = None

        self.ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        self.st = U.get_placeholder(name="st", dtype=tf.int32, shape=[None])

        if normalize_obs:
            with tf.variable_scope("obfilter"):
                self.ob_rms = RunningMeanStd(shape=ob_space.shape)
            if avg_norm_symmetry:
                # Warning works only for normal observations (41 numbers)
                ob_mean = (tf.gather(self.ob_rms.mean, ORIG_SYMMETRIC_IDS) + self.ob_rms.mean) / 2
                ob_std = (tf.gather(self.ob_rms.std, ORIG_SYMMETRIC_IDS) + self.ob_rms.std) / 2  # Pretty crude
            else:
                ob_mean = self.ob_rms.mean
                ob_std = self.ob_rms.std

            obz = tf.clip_by_value((self.ob - ob_mean) / ob_std, -stdclip, stdclip)

            #obz = tf.Print(obz, [self.ob_rms.mean], message='rms_mean', summarize=41)
            #obz = tf.Print(obz, [self.ob_rms.std], message='rms_std', summarize=41)
        else:
            obz = self.ob

        vpreds = []
        pparams = []

        for part in range(1 if not three else 3):
            part_prefix = "" if part == 0 else "part_" + str(part)

            # Predicted value
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(U.dense(last_out, hid_size, part_prefix + "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))

            vpreds.append(U.dense(last_out, 1, part_prefix + "vffinal", weight_init=U.normc_initializer(1.0)))
            vpreds[-1] = vpreds[-1][:, 0]

            if parallel_value:
                last_out_2 = obz
                for i in range(pv_layers):
                    last_out_2 = tf.nn.tanh(U.dense(last_out_2, pv_hid_size, part_prefix + "pv_vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
                last_out_2 = U.dense(last_out_2, 1, part_prefix + "pv_vffinal", weight_init=U.normc_initializer(1.0))
                vpreds[-1] += last_out_2[:, 0]

            last_out = obz
            if activation == 'tanh': activation = tf.nn.tanh
            elif activation == 'relu': activation = tf.nn.relu
            for i in range(num_hid_layers):
                dense = U.dense(last_out, hid_size, part_prefix + "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0))
                last_out = activation(dense)

            if actions == 'gaussian':
                if gaussian_fixed_var:
                    mean = U.dense(last_out, pdtype.param_shape()[0]//2, part_prefix + "polfinal", U.normc_initializer(0.01))
                    if bound_by_sigmoid:
                        mean = tf.nn.sigmoid(mean * sigmoid_coef)
                    logstd = tf.get_variable(name=part_prefix + "logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                    logstd = mean * 0.0 + logstd
                else:
                    mean = U.dense(last_out, pdtype.param_shape()[0]//2, part_prefix + "polfinal", U.normc_initializer(0.01))
                    logstd = U.dense(last_out, pdtype.param_shape()[0]//2, part_prefix + "polfinal_2", U.normc_initializer(0.01))
                if gaussian_bias:
                    mean = mean + 0.5

                pdparam = U.concatenate([mean, logstd], axis=1)
            elif actions == 'beta':
                 pdparam = U.dense(last_out, pdtype.param_shape()[0], part_prefix + "beta_lastlayer", U.normc_initializer(0.01))
                 pdparam = tf.nn.softplus(pdparam)
            elif actions in ['bernoulli', 'binary']:
                if bound_by_sigmoid:
                    raise NotImplementedError("bound by sigmoid not implemented here")
                pdparam = U.dense(last_out, pdtype.param_shape()[0], part_prefix + "polfinal", U.normc_initializer(0.01))
            elif actions in ['cat_3']:
                pdparam = U.dense(last_out, pdtype.param_shape()[0], part_prefix + "cat3_lastlayer", U.normc_initializer(0.01))
                # prob = tf.reshape(pdparam, [18, -1])
                # prob = tf.nn.softmax(prob)
                # elogit = tf.exp(pdparam)
                # pdparam = tf.Print(pdparam, [prob], summarize=18)
            elif actions in ['cat_5']:
                pdparam = U.dense(last_out, pdtype.param_shape()[0], part_prefix + "cat5_lastlayer", U.normc_initializer(0.01))
                # prob = tf.reshape(pdparam, [18, -1])
                # prob = tf.nn.softmax(prob)
                # elogit = tf.exp(pdparam)
                # pdparam = tf.Print(pdparam, [prob], summarize=18)
            else:
                assert False

            pparams.append(pdparam)

        pparams = tf.stack(pparams)
        vpreds = tf.stack(vpreds)
        pparams = tf.transpose(pparams, perm=(1, 0, 2))  # [batchsize, networks, values]
        vpreds = tf.transpose(vpreds, perm=(1, 0))  # [batchsize, networks, values]

        self.stochastic = tf.placeholder(name="stochastic", dtype=tf.bool, shape=())

        if three:
            batchsize = tf.shape(pdparam)[0]
            NO_OBSTACLES_ID = 5
            OBST_DIST = [278, 279, 280, 281, 282, 283, 284, 285]  # TODO: Alternative approach
            distances = [self.ob[:, i] for i in OBST_DIST]
            distances = tf.stack(distances, axis=1)
            no_obstacles = tf.cast(tf.equal(self.ob[:, NO_OBSTACLES_ID], 1.0), tf.int32)
            distances = tf.cast(tf.reduce_all(tf.equal(distances, 3), axis=1), tf.int32)
            no_obstacles_ahead = distances * no_obstacles  # 0 if obstacles, 1 if no obstacles
            begin = tf.cast(tf.less(self.st, 75), tf.int32)
            take_id = (1 - begin) * (1 + no_obstacles_ahead)  # begin==1 => 0, begin==0 => 1 + no_obstacles_ahead

            take_id = tf.stack((tf.range(batchsize), take_id), axis=1)
            pdparam = tf.gather_nd(pparams, take_id)

            self.vpred = tf.gather_nd(vpreds, take_id)
            #self.vpred = tf.Print(self.vpred, [take_id])
        else:
            self.vpred = vpreds[:, 0]
            pdparam = pparams[:, 0]

        self.pd = pdtype.pdfromflat(pdparam)

        if hasattr(self.pd, 'real_mean'):
            real_mean = self.pd.real_mean()
            ac = U.switch(self.stochastic, self.pd.sample(), real_mean)
        else:
            ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([self.stochastic, self.ob, self.st], [ac, self.vpred, ob_mean, ob_std])

        if actions == 'binary':
            self._binary_f = U.function([self.stochastic, self.ob, self.st], [ac, self.pd.flat, self.vpred])


    def act(self, stochastic, ob, t):
        ac1, vpred, ob_mean, ob_std = self._act(stochastic, ob[None], t[None])
        return ac1[0], vpred[0]

    def get_binary(self, stochastic, ob, t):
        return self._binary_f(stochastic, ob, t)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_orig_variables(self):
        return [x for x in self.get_variables() if 'part' not in x.name and 'obfilter' not in x.name]

    def get_part_variables(self, part):
        return [x for x in self.get_variables() if 'part_' + str(part) in x.name and 'obfilter' not in x.name]

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

