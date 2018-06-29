from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from collections import OrderedDict

def afix(a, actions):
    if actions == 'cat_3':
        return a / 2.
    elif actions == 'cat_5':
        return a / 4.
    return a

def evaluate(env, policy_func, *, load_model_path, n_episodes, stochastic, actions, execute_just=None):
    # TODO: Make it MPI and get rid of hacks in traj_segment_generator
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy

    U.initialize()
    U.load_state(load_model_path)

    #for v in pi.get_trainable_variables():
    #    assign_op = v.assign_add(tf.random_uniform(tf.shape(v), 0.0, 0.2))
    #    assign_op.eval()
    orig_scores = []
    shaped_scores = []
    delta_x_scores = []
    ligament_scores = []
    ep_lengths = []
    for episodeno in range(n_episodes):
        shaped_score = 0.
        orig_score = 0.
        ligament_score = 0.
        delta_x_score = 0.
        done = False
        obs = env.reset()
        if execute_just is not None and episodeno != execute_just:
            continue

        if obs is None:
            return
        t = 0
        while not done:
            a, _ = pi.act(stochastic, obs, np.int32(t))
            obs, r, done, info = env.step(afix(a, actions))
            shaped_score += r
            orig_score += info['original_reward']
            delta_x_score += info.get('delta_x_reward', 0)
            ligament_score += info.get('ligament_reward', 0)
            t += 1
        print()
        print(f'original  score: {orig_score:5.1f}')
        print(f'  - delta_x  score: {delta_x_score:6.3f}')
        print(f'  - ligament score: {ligament_score:6.3f}')
        print(f'shaped    score: {shaped_score:5.1f}')
        print(f'episode len    : {t:5.0f}')
        orig_scores.append(orig_score)
        delta_x_scores.append(delta_x_score)
        ligament_scores.append(ligament_score)
        shaped_scores.append(shaped_score)
        ep_lengths.append(t)

        if execute_just is not None:
            break

    print(f'avg(original score): {np.mean(orig_scores):5.1f}')
    print(f'  - avg(delta_x  score): {np.mean(delta_x_scores):5.1f}')
    print(f'  - avg(ligament score): {np.mean(ligament_scores):5.1f}')
    print(f'avg(shaped   score): {np.mean(shaped_scores):5.1f}')
    print(f'avg(len)           : {np.mean(ep_lengths):5.1f}')

seeds = []

def traj_segment_generator(pi, env, horizon, stochastic, single_episode=False, actions=None,
                           bootstrap_seeds=False):
    global seeds
    easy_seeds = []
    hard_seeds = []
    if bootstrap_seeds:
        wrapenv = env
        while not hasattr(wrapenv, 'inject_seed'):
            wrapenv = wrapenv.env
        if len(seeds) > 0:
            if np.random.rand() < 0.8:
                current_seed = np.random.choice(seeds)
            else:
                current_seed = wrapenv.np_random.randint(0, 2**32)
        else:
            current_seed = wrapenv.np_random.randint(0, 2**32)
        wrapenv.inject_seed(current_seed)

    t = 0
    ac = np.zeros(len(env.action_space.low))  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    stepno = 0

    cur_ep_ret = 0  # return in current episode
    cur_ep_orig_ret = 0
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_orig_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    steps = np.zeros(horizon, 'int32')
    rews = np.zeros(horizon, 'float32')
    orig_rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    done_maxsteps = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    nextvpreds = np.zeros(horizon, 'float32')
    nextisterminal = np.zeros(horizon, 'float32')

    juststarted = True

    i = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob, np.int32(stepno))  # get action to make and v(current_state)
        nextvpreds[i] = vpred
        if new and t > 0:
            ac2, vpred2 = pi.act(stochastic, lastob, np.int32(stepno))  # get action to make and v(current_state)
            nextvpreds[i] = vpred2

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if (t > 0 and t % horizon == 0) or (single_episode and not juststarted and new):
            if bootstrap_seeds:
                yield {"ob": obs, "step": steps, "rew": rews, "orig_rew": orig_rews, "vpred": vpreds, "new": news,
                       "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                       "nextvpreds": nextvpreds, "nextisterminal": nextisterminal,
                       "ep_rets": ep_rets, "ep_orig_rets": ep_orig_rets, "ep_lens": ep_lens,
                       "done_maxsteps": done_maxsteps, "easy_seeds": easy_seeds, "hard_seeds": hard_seeds}
            else:
                yield {"ob": obs, "step": steps, "rew": rews, "orig_rew": orig_rews, "vpred": vpreds, "new": news,
                       "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                       "ep_rets": ep_rets, "ep_orig_rets": ep_orig_rets, "ep_lens": ep_lens,
                       "nextvpreds": nextvpreds, "nextisterminal": nextisterminal,
                       "done_maxsteps": done_maxsteps}
            # Comments:
            #   H = horizon
            #   ob[0], ob[1], .. ob[H-1]
            #   vpreds[0],... vpreds[H-1]
            #   rew[0], rew[1],..., res[H-1]
            #   new[0], new[1],..., new[H-1]  (new[t]==True if the previous reward was the last before a terminal state)
            #   nextvpred = vpreds[H] if H is not a terminal state

            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_orig_rets = []
            ep_lens = []
            easy_seeds = []
            hard_seeds = []

        i = t % horizon
        obs[i] = ob
        steps[i] = stepno
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, info = env.step(afix(ac, actions))
        lastob = ob
        stepno += 1
        rews[i] = rew
        orig_rews[i] = info['original_reward']
        done_maxsteps[i] = info['done_maxsteps']
        nextisterminal[i] = new

        cur_ep_ret += rew
        cur_ep_orig_ret += info['original_reward']
        cur_ep_len += 1
        if new:
            # Notice: the last observation (ob) is never used anywhere. It is overriden by ob=env.reset()
            if bootstrap_seeds:
                if cur_ep_len == 1000:
                    easy_seeds.append(current_seed)
                    hard_seeds.append(0)
                elif cur_ep_len < 1000 and info['original_reward'] != -0.045 and 'Timeouted' not in info:
                    hard_seeds.append(current_seed)
                    easy_seeds.append(0)
                else:
                    hard_seeds.append(0)
                    easy_seeds.append(0)
                if len(set(seeds) - set(easy_seeds)) > 0:
                    current_seed = np.random.choice(list(set(seeds) - set(easy_seeds)))
                else:
                    current_seed = wrapenv.np_random.randint(0, 2**32)
                wrapenv.inject_seed(current_seed)

            ep_rets.append(cur_ep_ret)
            ep_orig_rets.append(cur_ep_orig_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_orig_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            stepno = 0
            if single_episode:
                t = -1
                juststarted = False
        t += 1


def add_vtarg_and_adv(seg, gamma, lam, horizon_hack):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t+1]
        if horizon_hack:
            if seg['nextisterminal'][t] and not seg['done_maxsteps'][t]:
                futurev = 0
            else:  # i.e. not seg[nextisterminal][t] or seg[done_maxsteps']
                futurev = seg["nextvpreds"][t]
            delta = rew[t] + gamma * futurev - vpred[t]
        else:
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def swap_legs(obs_dict):
    def get_left(key):
        return key.replace('_r_', '_l_')

    def swap(a, b):
        obs_dict[a], obs_dict[b] = obs_dict[b], obs_dict[a]

    keys = obs_dict.keys()
    for key in keys:
        if '_r_' in key:
            swap(key, get_left(key))
    swap('strength_l', 'strength_r')


# noinspection SpellCheckingInspection
def learn(env, policy_func, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          load_model_path,
          test_only,
          stochastic,
          symmetric_training=False,
          obs_names=None,
          single_episode=False,
          horizon_hack=False,
          running_avg_len=100,
          init_three=False,
          actions=None,
          symmetric_training_trick=False,
          seeds_fn=None,
          bootstrap_seeds=False,
          ):
    global seeds
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)                 # Network for new policy
    old_pi = policy_func("old_pi", ob_space, ac_space)         # Network for old policy
    adv_targ = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])       # Empirical return
    mask = tf.placeholder(dtype=tf.bool, shape=[None])      # Mask for the trick

    lr_mult = tf.placeholder(name='lr_mult', dtype=tf.float32, shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lr_mult                                     # Annealed clipping parameter epsilon

    ob = U.get_placeholder_cached(name="ob")
    st = U.get_placeholder_cached(name="st")
    ac = pi.pdtype.sample_placeholder([None])

    kl = old_pi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    mean_kl = U.mean(tf.boolean_mask(kl, mask))  # Mean over the batch
    mean_ent = U.mean(tf.boolean_mask(ent, mask))
    entropy_penalty = -entcoeff * mean_ent

    ratio = tf.exp(pi.pd.logp(ac) - old_pi.pd.logp(ac))  # pi_new / pi_old
    surr_1 = ratio * adv_targ  # surrogate from conservative policy iteration
    surr_2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ  #
    surr_loss = -U.mean(tf.boolean_mask(tf.minimum(surr_1, surr_2), mask))  # PPO's pessimistic surrogate (L^CLIP), mean over the batch
    vf_loss = U.mean(tf.boolean_mask(tf.square(pi.vpred - ret), mask))
    total_loss = surr_loss + entropy_penalty + vf_loss
    losses = [surr_loss, entropy_penalty, vf_loss, mean_kl, mean_ent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    comp_loss_and_grad = U.function([ob, st, ac, adv_targ, ret, lr_mult, mask], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[
        tf.assign(old_v, new_v) for (old_v, new_v) in zipsame(old_pi.get_variables(), pi.get_variables())])
    comp_loss = U.function([ob, st, ac, adv_targ, ret, lr_mult, mask], losses)

    if init_three:
        assign_init_three_1 = U.function([], [], updates=[
            tf.assign(new_v, old_v) for (old_v, new_v) in zipsame(pi.get_orig_variables(), pi.get_part_variables(1))])
        assign_init_three_2 = U.function([], [], updates=[
            tf.assign(new_v, old_v) for (old_v, new_v) in zipsame(pi.get_orig_variables(), pi.get_part_variables(2))])

    U.initialize()
    if load_model_path is not None:
        U.load_state(load_model_path)
        if init_three:
            assign_init_three_1()
            assign_init_three_2()
    adam.sync()

    if seeds_fn is not None:
        with open(seeds_fn) as f:
            seeds = [int(seed) for seed in f.readlines()]
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=stochastic, single_episode=test_only or single_episode, actions=actions, bootstrap_seeds=bootstrap_seeds)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    len_buffer = deque(maxlen=running_avg_len)      # rolling buffer for episode lengths
    rew_buffer = deque(maxlen=running_avg_len)      # rolling buffer for episode rewards
    origrew_buffer = deque(maxlen=running_avg_len)  # rolling buffer for original episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0, max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
 
        add_vtarg_and_adv(seg, gamma, lam, horizon_hack=horizon_hack)

        # ob, ac, adv_targ, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, st, ac, adv_targ, tdlamret = seg["ob"], seg["step"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate

        if symmetric_training_trick:
            first_75 = st < 75
            mask = ~np.concatenate((np.zeros_like(first_75), first_75))
        else:
            mask = np.concatenate((np.ones_like(st, dtype=np.bool), np.ones_like(st, dtype=np.bool)))
        if symmetric_training:
            sym_obss = []
            sym_acc = []
            for i in range(timesteps_per_batch):
                obs = OrderedDict(zip(obs_names, ob[i]))
                sym_obs = obs.copy()
                swap_legs(sym_obs)

                sym_ac = ac[i].copy()
                sym_ac = np.concatenate((sym_ac[9:], sym_ac[:9]))
                sym_obss.append(np.asarray(list(sym_obs.values())))
                sym_acc.append(sym_ac)
            sym_obss = np.asarray(sym_obss)
            sym_acc = np.asarray(sym_acc)

            ob = np.concatenate((ob, sym_obss))
            ac = np.concatenate((ac, sym_acc))
            adv_targ = np.concatenate((adv_targ, adv_targ))
            tdlamret = np.concatenate((tdlamret, tdlamret))
            vpredbefore = np.concatenate((vpredbefore, vpredbefore))
            st = np.concatenate((st, st))

        # Compute stats before updating
        if bootstrap_seeds:
            lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_orig_rets"], seg["easy_seeds"], seg["hard_seeds"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews, orig_rews, easy_seeds, hard_seeds = map(flatten_lists, zip(*listoflrpairs))
            easy_seeds = [x for x in easy_seeds if x != 0]
            hard_seeds = [x for x in hard_seeds if x != 0]
            print('seeds set sizes:', len(seeds), len(easy_seeds), len(hard_seeds))
            seeds = list((set(seeds) - set(easy_seeds)) | set(hard_seeds))
        else:
            lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_orig_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews, orig_rews = map(flatten_lists, zip(*listoflrpairs))
 
        len_buffer.extend(lens)
        rew_buffer.extend(rews)
        origrew_buffer.extend(orig_rews)
        logger.record_tabular("Iter", iters_so_far)
        logger.record_tabular("EpLenMean", np.mean(len_buffer))
        logger.record_tabular("EpRewMean", np.mean(rew_buffer))
        logger.record_tabular("EpOrigRewMean", np.mean(origrew_buffer))
        logger.record_tabular("EpOrigRewStd", np.std(origrew_buffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        n_completed = 0
        sum_completed = 0
        for ep_len, orig_rew in zip(lens, orig_rews):
            if ep_len == 1000:
                n_completed += 1
                sum_completed += orig_rew
        avg_completed = sum_completed / n_completed if n_completed > 0 else 0
        logger.record_tabular("AvgCompleted", avg_completed)
        perc_completed = 100 * n_completed / len(lens) if len(lens) > 0 else 0
        logger.record_tabular("PercCompleted", perc_completed)

        if callback: callback(locals(), globals())

        adv_targ = (adv_targ - adv_targ.mean()) / adv_targ.std()  # standardized advantage function estimate
        d = Dataset(dict(ob=ob, st=st, ac=ac, atarg=adv_targ, vtarg=tdlamret, mask=mask), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values
        logger.log("Optimizing...")
        if not test_only:
            logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data. I log results only for the first worker (rank=0)
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *batch_losses, grads = comp_loss_and_grad(batch["ob"], batch["st"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, batch["mask"])
                if not test_only:
                    adam.update(grads, optim_stepsize * cur_lrmult)
                losses.append(batch_losses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            batch_losses = comp_loss(batch["ob"], batch["st"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, batch["mask"])
            losses.append(batch_losses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

        iters_so_far += 1


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
