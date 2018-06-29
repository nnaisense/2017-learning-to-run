#!/usr/bin/env python
import argparse
import os
import shutil
import socket
import sys
import time
from os import path
from pathlib import Path

import gym
import logging
import numpy as np
from mpi4py import MPI

from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.mpi_fork import mpi_fork
from turnips.walker import Walker, MuscleWalker, RepeatActionsWalker, RunEnvWrapper, SubmitRunEnv, h5pyEnvLogger
from turnips.MyRunEnv import IsolatedMyRunEnv


def submit_round2(walker_env, submit_env, policy_fn, load_model_path, stochastic, actions):
    ob_space = walker_env.observation_space
    ac_space = walker_env.action_space

    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy

    U.initialize()
    U.load_state(load_model_path)

    while True:
        obs = walker_env.reset()
        stepno = 0
        if isinstance(obs, bool) and obs == False:
            break
        done = False
        while not done:
            action, _ = pi.act(stochastic, obs, np.int32(stepno))
            obs, rew, done, info = walker_env.step(action)
            stepno += 1
            if done:
                break

    submit_env.submit()


def train(args):
    from baselines.pposgd import mlp_policy, pposgd_simple
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    logger.session(dir=args.exp_path, format_strs=None if rank == 0 and not args.test_only and not args.evaluate else []).__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = args.seed + 10000 * rank
    set_global_seeds(workerseed)

    if args.submit:
        env = SubmitRunEnv(visualize=args.render)
    elif args.submit_round2:
        from turnips.submit_round2_env import SubmitRunEnv2
        submit_env = env = SubmitRunEnv2()
    elif args.simwalker:
        env = SimWalker(visualize=args.render)
    else:
        env = IsolatedMyRunEnv(visualize=args.render, run_logs_dir=args.run_logs_dir, additional_info={'exp_name': args.exp_name}, step_timeout=args.step_timeout,
                               n_obstacles=args.n_obstacles, higher_pelvis=args.higher_pelvis)

    env = RunEnvWrapper(env, args.diff)
    if args.simwalker and args.log_simwalker:
        cls = type("h5pyEnvLoggerClone", (gym.Wrapper,), dict(h5pyEnvLogger.__dict__))  # workaround for double wrap problem
        env = cls(env, log_dir=args.run_logs_dir, filename_prefix='simwalker_',
            additional_info={'exp_name': args.exp_name, 'difficulty': args.diff, 'seed': args.seed})

    env = env_walker = Walker(env, shaping_mode=args.shaping, transform_inputs=args.transform_inputs,
                              obstacle_hack=not args.noobsthack, max_steps=args.max_env_steps,
                              memory_size=args.memory_size, swap_legs_mode=args.swap_legs_mode,
                              filter_obs=args.filter_obs, add_time=args.add_time, fall_penalty=args.fall_penalty,
                              fall_penalty_value=args.fall_penalty_val, print_action=args.print_action,
                              new8_fix=args.new8_fix, pause=args.pause, noisy_obstacles=args.noisy_obstacles, noisy_obstacles2=args.noisy_obstacles2,
                              noisy_fix=args.noisy_fix)

    if args.log_walker:
        env = h5pyEnvLogger(env, log_dir=args.run_logs_dir, filename_prefix='walker_',
            additional_info={'exp_name': args.exp_name, 'difficulty': args.diff, 'seed': args.seed})
    if args.muscles:
        env = MuscleWalker(env)
    if args.repeats > 1:
        env = RepeatActionsWalker(env, args.repeats)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=args.hid_size, num_hid_layers=args.num_hid_layers,
                                    bound_by_sigmoid=args.bound_by_sigmoid,
                                    sigmoid_coef=args.sigmoid_coef,
                                    activation=args.activation,
                                    normalize_obs=not args.nonormalize_obs,
                                    gaussian_fixed_var=not args.nogaussian_fixed_var,
                                    avg_norm_symmetry=args.avg_norm_symmetry,
                                    symmetric_interpretation=args.symmetric_interpretation,
                                    stdclip=args.stdclip, actions=args.actions,
                                    gaussian_bias=args.gaussian_bias,
                                    gaussian_from_binary=args.gaussian_from_binary,
                                    parallel_value=args.parallel_value, pv_layers=args.pv_layers, pv_hid_size=args.pv_hid_size,
                                    three=args.three)

    if not args.test_only and not args.evaluate:
        env = bench.Monitor(env, path.join(args.exp_path, "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    current_best = float('-inf')
    current_best_completed = float('-inf')
    current_best_perc_completed = float('-inf')
    stats_f = None

    start = time.time()

    def callback(local_, global_):
        nonlocal current_best
        nonlocal current_best_completed
        nonlocal current_best_perc_completed
        nonlocal stats_f
        if rank != 0: return
        if args.test_only or args.evaluate: return

        print('ELAPSED', time.time() - start)
        print(f'{socket.gethostname()}:{args.exp_path}')

        iter_no = local_['iters_so_far']
        if iter_no % args.save_every == 0:
            U.save_state(path.join(args.exp_path, 'models', f'{iter_no:04d}', 'model'))

        if local_['iters_so_far'] == 0:
            stats_f = open(path.join(args.exp_path, 'simple_stats.csv'), 'w')
            cols = ["Iter", "EpLenMean", "EpRewMean", "EpOrigRewMean", "EpThisIter", "EpisodesSoFar", "TimestepsSoFar", "TimeElapsed", "AvgCompleted", "PercCompleted"]
            for name in local_['loss_names']: cols.append("loss_" + name)
            stats_f.write(",".join(cols) + '\n')
        else:
            current_orig_reward = np.mean(local_['origrew_buffer'])
            if current_best < current_orig_reward:
                print(f'Found better {current_best:.2f} -> {current_orig_reward:.2f}')
                current_best = current_orig_reward
                U.save_state(path.join(args.exp_path, 'best', 'model'))
            U.save_state(path.join(args.exp_path, 'last', 'model'))

            avg_completed = local_["avg_completed"]
            if current_best_completed < avg_completed:
                print(f'Found better completed {current_best_completed:.2f} -> {avg_completed:.2f}')
                current_best_completed = avg_completed
                U.save_state(path.join(args.exp_path, 'best_completed', 'model'))

            perc_completed = local_["perc_completed"]
            if current_best_perc_completed < perc_completed:
                print(f'Found better perc completed {current_best_perc_completed:.2f} -> {perc_completed:.2f}')
                current_best_perc_completed = perc_completed
                U.save_state(path.join(args.exp_path, 'perc_completed', 'model'))

            data = [
                local_['iters_so_far'],
                np.mean(local_['len_buffer']),
                np.mean(local_['rew_buffer']),
                np.mean(local_['origrew_buffer']),
                len(local_['lens']),
                local_['episodes_so_far'],
                local_['timesteps_so_far'],
                time.time() - local_['tstart'],
                avg_completed,
                perc_completed,
            ]
            if 'meanlosses' in local_:
                for lossval in local_['meanlosses']:
                    data.append(lossval)

            stats_f.write(",".join([str(x) for x in data]) + '\n')
            stats_f.flush()

    if args.load_model is not None:
        args.load_model += '/model'
    if args.submit_round2:
        submit_round2(env, submit_env, policy_fn, load_model_path=args.load_model, stochastic=False, actions=args.actions)
        #submit_env.submit()   # submit_round2(...) submits already
        sys.exit()
    if args.evaluate:
        pposgd_simple.evaluate(env, policy_fn, load_model_path=args.load_model, n_episodes=args.n_eval_episodes,
                               stochastic=not args.nostochastic, actions=args.actions, execute_just=args.execute_just)
    else:
        pposgd_simple.learn(
            env, policy_fn,
            max_timesteps=args.max_timesteps,
            timesteps_per_batch=args.timesteps_per_batch,
            clip_param=args.clip_param, entcoeff=args.entcoeff,
            optim_epochs=args.optim_epochs, optim_stepsize=args.optim_stepsize, optim_batchsize=args.optim_batchsize,
            gamma=args.gamma, lam=args.lam,
            callback=callback,
            load_model_path=args.load_model,
            test_only=args.test_only,
            stochastic=not args.nostochastic,
            symmetric_training=args.symmetric_training,
            obs_names=env_walker.obs_names,
            single_episode=args.single_episode,
            horizon_hack=args.horizon_hack,
            running_avg_len=args.running_avg_len,
            init_three=args.init_three,
            actions=args.actions,
            symmetric_training_trick=args.symmetric_training_trick,
            bootstrap_seeds=args.bootstrap_seeds,
            seeds_fn=args.seeds_fn,
        )
    env.close()


def prepare_env(args):
    if path.exists(args.exp_path):
        if args.force_override or args.exp_name == 'tmp':
            print('remove')
            shutil.rmtree(args.exp_path, ignore_errors=True)
        else:
            raise Exception('The experiment dir already exists. Consider force_override')
    os.makedirs(args.exp_path, exist_ok=True)

    # Save command
    with open(path.join(args.exp_path, 'command'), "w") as f:
        cmd = [path.relpath(sys.argv[0])] + sys.argv[1:]
        f.write(" ".join(cmd) + "\n\n")
        f.write(str(args))

    # Save sources
    sources_dir = path.join(args.exp_path, 'src')
    shutil.rmtree(sources_dir, ignore_errors=True)
    os.makedirs(sources_dir, exist_ok=True)
    time.sleep(1)
    try:
        for source_file in ['run_walker.py', 'baselines']:
            if path.isdir(source_file):
                ignore_func = lambda d, files: [f for f in files if
                                                (Path(d) / Path(f)).is_file() and not f.endswith('.py')]
                shutil.copytree(source_file, path.join(sources_dir, source_file), ignore=ignore_func)
            else:
                shutil.copyfile(source_file, path.join(sources_dir, source_file))
    except e:
        print('Some src copytree error')


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-n', '--exp_name', dest='exp_name', default='tmp')
    parser.add_argument('-r', '--render', dest='render', action='store_true')
    parser.add_argument('-c', '--num_cpu', dest='num_cpu', default=1, type=int)
    parser.add_argument('--resdir', dest='resdir', default='results')
    parser.add_argument('--max_timesteps', dest='max_timesteps', default=1e9, type=int)
    parser.add_argument('--seed', dest='seed', default=123, type=int)
    parser.add_argument('--force_override', dest='force_override', action='store_true')
    parser.add_argument('--timesteps_per_batch', dest='timesteps_per_batch', default=2048, type=int)
    parser.add_argument('--clip_param', dest='clip_param', default=0.2, type=float)
    parser.add_argument('--optim_epochs', dest='optim_epochs', default=10, type=int)
    parser.add_argument('--optim_stepsize', dest='optim_stepsize', default=3e-4, type=float)
    parser.add_argument('--optim_batchsize', dest='optim_batchsize', default=64, type=int)
    parser.add_argument('--entcoeff', dest='entcoeff', default=0., type=float)
    parser.add_argument('--gamma', dest='gamma', default=0.99, type=float)
    parser.add_argument('--lam', dest='lam', default=0.95, type=float)
    parser.add_argument('--hid_size', dest='hid_size', default=64, type=int)
    parser.add_argument('--num_hid_layers', dest='num_hid_layers', default=2, type=int)
    parser.add_argument('--shaping', dest='shaping', default=None, type=str)
    parser.add_argument('--save_every', dest='save_every', default=20, type=int)
    parser.add_argument('--diff', dest='diff', default=0, type=int)
    parser.add_argument('--relative_x', dest='relative_x', action='store_true', help='DEPRECATED')
    parser.add_argument('--transform_inputs', dest='transform_inputs', type=str, default=None)
    parser.add_argument('--bound_by_sigmoid', dest='bound_by_sigmoid', action='store_true')
    parser.add_argument('--sigmoid_coef', dest='sigmoid_coef', default=1., type=float)
    parser.add_argument('--noobsthack', dest='noobsthack', action='store_true')
    parser.add_argument('--nogaussian_fixed_var', dest='nogaussian_fixed_var', action='store_true')

    parser.add_argument('--activation', dest='activation', default='tanh', type=str)
    parser.add_argument('--nonormalize_obs', dest='nonormalize_obs', action='store_true')

    parser.add_argument('--nostochastic', dest='nostochastic', action='store_true')
    parser.add_argument('--nostochastic2', dest='nostochastic2', action='store_true')
    parser.add_argument('--load_model', dest='load_model', default=None, type=str)
    parser.add_argument('--test_only', dest='test_only', action='store_true')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--n_eval_episodes', dest='n_eval_episodes', default=10000, type=int)
    parser.add_argument('--submit', dest='submit', action='store_true')
    parser.add_argument('--max_env_steps', dest='max_env_steps', default=1000, type=int)
    parser.add_argument('--run_logs_dir', dest='run_logs_dir', default=None, type=str)
    parser.add_argument('--avg_norm_symmetry', dest='avg_norm_symmetry', action='store_true')
    parser.add_argument('--symmetric_interpretation', dest='symmetric_interpretation', action='store_true')
    parser.add_argument('--stdclip', dest='stdclip', default=5.0, type=float)
    parser.add_argument('--memory_size', dest='memory_size', default=1, type=int)
    parser.add_argument('--swap_legs_mode', dest='swap_legs_mode', default=None, type=str)
    parser.add_argument('--filter_obs', dest='filter_obs', action='store_true')
    parser.add_argument('--actions', dest='actions', default='gaussian', type=str)

    parser.add_argument('--binary_actions', dest='binary_actions', action='store_true', help='deprecated')
    parser.add_argument('--beta_dist', dest='beta_dist', action='store_true', help='deprecated')
    parser.add_argument('--gaussian_bias', dest='gaussian_bias', action='store_true')
    parser.add_argument('--muscles', dest='muscles', action='store_true')
    parser.add_argument('--repeats', dest='repeats', default=1, type=int)
    parser.add_argument('--add_time', dest='add_time', action='store_true')
    parser.add_argument('--simwalker', dest='simwalker', action='store_true')
    parser.add_argument('--log_walker', dest='log_walker', action='store_true')
    parser.add_argument('--log_simwalker', dest='log_simwalker', action='store_true')
    parser.add_argument('--symmetric_training', dest='symmetric_training', action='store_true')
    parser.add_argument('--step_timeout', dest='step_timeout', default=None, type=float)
    parser.add_argument('--gaussian_from_binary', dest='gaussian_from_binary', action='store_true')
    parser.add_argument('--pv', dest='parallel_value', action='store_true')
    parser.add_argument('--pv_layers', dest='pv_layers', default=2, type=int)
    parser.add_argument('--pv_hid_size', dest='pv_hid_size', default=512, type=int)
    parser.add_argument('--horizon_hack', dest='horizon_hack', action='store_true')
    parser.add_argument('--single_episode', dest='single_episode', action='store_true')
    parser.add_argument('--n_obstacles', dest='n_obstacles', default=3, type=int)
    parser.add_argument('--nologs', dest='nologs', action='store_true')
    parser.add_argument('--init_three', dest='init_three', action='store_true')
    parser.add_argument('--three', dest='three', action='store_true')
    parser.add_argument('--pause', dest='pause', action='store_true')
    parser.add_argument('--nobind', dest='nobind', action='store_true')
    parser.add_argument('--running_avg_len', dest='running_avg_len', default=100, type=int)
    parser.add_argument('--submit_token', dest='submit_token', default=None, type=str)
    parser.add_argument('--fall_penalty', dest='fall_penalty', action='store_true')
    parser.add_argument('--fall_penalty_val', dest='fall_penalty_val', default=2., type=float)
    parser.add_argument('--higher_pelvis', dest='higher_pelvis', default=0.65, type=float)
    parser.add_argument('--print_action', dest='print_action', action='store_true')
    parser.add_argument('--new8_fix', dest='new8_fix', action='store_true')
    parser.add_argument('--symmetric_training_trick', dest='symmetric_training_trick', action='store_true')
    parser.add_argument('--submit_round2', dest='submit_round2', action='store_true')
    parser.add_argument('--noisy_obstacles', dest='noisy_obstacles', action='store_true')
    parser.add_argument('--noisy_obstacles2', dest='noisy_obstacles2', action='store_true')
    parser.add_argument('--execute_just', dest='execute_just', default=None, type=int)
    parser.add_argument('--seeds_fn', dest='seeds_fn', default=None, type=str)
    parser.add_argument('--bootstrap_seeds', dest='bootstrap_seeds', action='store_true')
    parser.add_argument('--noisy_fix', dest='noisy_fix', action='store_true')

    args = parser.parse_args()

    if args.transform_inputs in ['new_5', 'new_6', 'new_7', 'new_8', 'new_9', 'new_a', 'new_8b']:
        args.filter_obs = True

    if args.binary_actions:
        logger.warn('Deprecated option')
        args.actions = 'binary'
    if args.beta_dist:
        logger.warn('Deprecated option')
        args.actions = 'beta'

    if args.relative_x:
        assert args.transform_inputs is None
        args.transform_inputs = 'relative_x'

    if args.transform_inputs == 'new_4':
        logger.warn("Overriding the memory size to 3")
        args.memory_size = 3

    if args.submit:
        assert args.load_model
        args.evaluate = True

    if args.submit_round2:
        assert args.load_model
        args.evaluate = True
        args.n_eval_episodes = 100000
        args.log_simwalker = False
        args.log_walker = False
        args.nobind = True
        args.num_cpu = 1
        args.nologs = True

    if args.render:
        args.num_cpu = 1

    # Create exp dir
    env_name = f'Walker_d{args.diff}'
    if args.max_env_steps is not None and args.max_env_steps != 1000:
        env_name += f'_{args.max_env_steps:03d}'
    if args.n_obstacles != 3:
        env_name += f'_o{args.n_obstacles:02d}'
    env_name += '-v0'

    args.exp_path = path.join(args.resdir, env_name, 'PPOOAI', args.exp_name, str(args.seed))
    if args.run_logs_dir is None and not args.test_only and not args.evaluate:
        args.run_logs_dir = path.join(args.exp_path, 'run_logs')
    if args.nologs:
        args.run_logs_dir = None

    whoami = mpi_fork(args.num_cpu, not args.nobind)
    if whoami == 'parent': return
    if MPI.COMM_WORLD.Get_rank() == 0:
        if not args.test_only and not args.evaluate:
            prepare_env(args)
    else:
        time.sleep(0.5)  # Just in case

    train(args)


if __name__ == '__main__':
    main()
