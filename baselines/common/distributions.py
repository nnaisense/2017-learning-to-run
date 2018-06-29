import tensorflow as tf
import numpy as np
import baselines.common.tf_util as U
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError
    def logp(self, x):
        return - self.neglogp(x)

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32


class MultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return tf.int32

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd
    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class BetaPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.size = len(low)
    def pdclass(self):
        return BetaPd
    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return BernoulliPd
    def param_shape(self):
        return [self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.int32

# WRONG SECOND DERIVATIVES
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.nn.softmax(logits)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return U.argmax(self.logits, axis=1)
#     def logp(self, x):
#         return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)
#     def kl(self, other):
#         return tf.nn.softmax_cross_entropy_with_logits(other.logits, self.ps) \
#                 - tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def entropy(self):
#         return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.argmax(self.logits - tf.log(-tf.log(u)), axis=1)

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        # self.logits = tf.Print(self.logits, [self.logits, tf.reduce_sum(1 / (1 + tf.exp(self.logits[:]))), U.argmax(self.logits, axis=1)])
        # self.logits = tf.Print(self.logits, [tf.abs(1 / (1 + tf.exp(self.logits[:])) - 0.5)])
        return U.argmax(self.logits, axis=1)
    def neglogp(self, x):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(Pd):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = tf.constant(low, dtype=tf.int32)
        self.categoricals = list(map(CategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.low + tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
    def kl(self, other):
        return tf.add_n([
                p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
            ])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        return self.low + tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.get_shape()) - 1, num_or_size_splits=2, value=flat)
        #mean = tf.Print(mean, [mean], summarize=18, message='mean')
        #logstd = tf.Print(logstd, [tf.exp(logstd)], summarize=18, message='std')
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean

    # This works only for [0, 1] range
    def real_mean(self):
        inpar = -tf.exp(-(-1 + self.mean)**2 / (2 * self.std**2)) + tf.exp(-self.mean**2 / (2 * self.std**2))
        a1 = inpar * self.std / np.sqrt(2 * np.pi)
        a2 = 0.5 * self.mean * tf.erf((1 - self.mean) / (np.sqrt(2) * self.std))
        a3 = 0.5 * self.mean * tf.erf(self.mean / (np.sqrt(2) * self.std))
        b = 0.5 * (1 + tf.erf((-1 + self.mean) / (np.sqrt(2) * self.std)))

        return a1 + a2 + a3 + b

    def neglogp(self, x):
        return 0.5 * U.sum(tf.square((x - self.mean) / self.std), axis=len(x.get_shape()) - 1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + U.sum(self.logstd, axis=len(x.get_shape()) - 1)
    def kl(self, other):
        # assert isinstance(other, DiagGaussianPd)
        return U.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
    def entropy(self):
        return U.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)
        # self.ps = tf.Print(self.ps, [self.ps], summarize=18, message='self.ps')
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.round(self.ps)
    def real_mean(self):
        return self.ps
    def neglogp(self, x):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=1)
    def kl(self, other):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=1) - U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=1)
    def entropy(self):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.ps))
        return tf.to_float(math_ops.less(u, self.ps))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


class BetaPd(Pd):
    def __init__(self, flat):
        self.min_value = 0.
        self.max_value = 1.

        self.flat = flat
        #flat = tf.Print(flat, [flat], message='flat', summarize=36)
        self.alpha, self.beta = tf.split(axis=len(flat.get_shape()) - 1, num_or_size_splits=2, value=flat)
        self.sum = self.alpha + self.beta
        self.mean = self.alpha / tf.maximum(x=self.sum, y=1e-6)
        #self.alpha = tf.Print(self.alpha, [self.alpha], message='alpha', summarize=18)
        #self.beta = tf.Print(self.beta, [self.beta], message='beta', summarize=18)
        #self.mean = tf.Print(self.mean, [self.mean], message='mean', summarize=18)
        self.log_norm = tf.lgamma(self.alpha) + tf.lgamma(self.beta) - tf.lgamma(self.sum)

    def flatparam(self):
        return self.flat
    def mode(self):
        return self.mean

    def neglogp(self, action):
        action = (action - self.min_value) / (self.max_value - self.min_value)
        action = tf.minimum(x=action, y=(1.0 - 1e-6))
        logp_ = (self.alpha - 1.0) * tf.log(action) + (self.beta - 1.0) * tf.log1p(-action) - self.log_norm
        #logp_ = tf.Print(logp_, [action[0,:2], self.alpha[0,:2], self.beta[0,:2], logp_[0,:2]], message='act,a,b,logp', summarize=18)
        return -U.sum(logp_, axis=-1)

    def kl(self, other):
        return U.sum(other.log_norm - self.log_norm - tf.digamma(self.beta) * (other.beta - self.beta) - \
               tf.digamma(self.alpha) * (other.alpha - self.alpha) + tf.digamma(self.sum) * (other.sum - self.sum), axis=-1)

    def entropy(self):
        e = self.log_norm - (self.beta - 1.0) * tf.digamma(self.beta) - \
               (self.alpha - 1.0) * tf.digamma(self.alpha) + ((self.sum - 2.0) * tf.digamma(self.sum))
        return U.sum(e, axis=-1)

    def sample(self):
        alpha_sample = tf.random_gamma(shape=(), alpha=self.alpha)
        beta_sample = tf.random_gamma(shape=(), alpha=self.beta)

        sample = alpha_sample / tf.maximum(x=(alpha_sample + beta_sample), y=1e-6)

        return self.min_value + sample * (self.max_value - self.min_value)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalPdType(ac_space.low, ac_space.high)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliPdType(ac_space.n)
    else:
        raise NotImplementedError

def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]

@U.in_session
def test_probtypes():
    np.random.seed(0)

    pdparam_diag_gauss = np.array([-.2, .3, .4, -.5, .1, -.5, .1, 0.8])
    diag_gauss = DiagGaussianPdType(pdparam_diag_gauss.size // 2) #pylint: disable=E1101
    validate_probtype(diag_gauss, pdparam_diag_gauss)

    pdparam_categorical = np.array([-.2, .3, .5])
    categorical = CategoricalPdType(pdparam_categorical.size) #pylint: disable=E1101
    validate_probtype(categorical, pdparam_categorical)

    pdparam_bernoulli = np.array([-.2, .3, .5])
    bernoulli = BernoulliPdType(pdparam_bernoulli.size) #pylint: disable=E1101
    validate_probtype(bernoulli, pdparam_bernoulli)


def validate_probtype(probtype, pdparam):
    N = 100000
    # Check to see if mean negative log likelihood == differential entropy
    Mval = np.repeat(pdparam[None, :], N, axis=0)
    M = probtype.param_placeholder([N])
    X = probtype.sample_placeholder([N])
    pd = probtype.pdclass()(M)
    calcloglik = U.function([X, M], pd.logp(X))
    calcent = U.function([M], pd.entropy())
    Xval = U.eval(pd.sample(), feed_dict={M:Mval})
    logliks = calcloglik(Xval, Mval)
    entval_ll = - logliks.mean() #pylint: disable=E1101
    entval_ll_stderr = logliks.std() / np.sqrt(N) #pylint: disable=E1101
    entval = calcent(Mval).mean() #pylint: disable=E1101
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = probtype.param_placeholder([N])
    pd2 = probtype.pdclass()(M2)
    q = pdparam + np.random.randn(pdparam.size) * 0.1
    Mval2 = np.repeat(q[None, :], N, axis=0)
    calckl = U.function([M, M2], pd.kl(pd2))
    klval = calckl(Mval, Mval2).mean() #pylint: disable=E1101
    logliks = calcloglik(Xval, Mval2)
    klval_ll = - entval - logliks.mean() #pylint: disable=E1101
    klval_ll_stderr = logliks.std() / np.sqrt(N) #pylint: disable=E1101
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr # within 3 sigmas

