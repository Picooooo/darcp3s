# Inspired from https://github.com/thu-ml/tianshou/blob/master/examples/mujoco/mujoco_sac.py


#!/usr/bin/env python3
import sys
import os
import gym
import torch
import pprint
import datetime
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from esrl.util import to_numpy, get_size, get_params, set_params
from esrl.esrl_trainer_v3 import esrl_trainer_v3
from esrl.trainer_v3 import trainer_v3
from esrl.ES import sepCEM, Control

from esrl.policy.td3 import P3STD3Policy
from tianshou.exploration import GaussianNoise
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.continuous import Actor, Critic
from esrl.data.collector import Collector, ReplayBuffer, VectorReplayBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--q_weight', type=float, default=0.1)
    parser.add_argument('--regularization_weight', type=float, default=0.005)
    # parser.add_argument('--alpha', type=float, default=0.2)
    # parser.add_argument('--auto-alpha', default=False, action='store_true')
    # parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # TD3 parameters
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)

    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    
    # ES parameters
    parser.add_argument('--pop_size', default=8, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')
    parser.add_argument('--max_step', default=1e6, type=int)
    parser.add_argument('--start-step',type=int, default=0)
    parser.add_argument('--episode-per-epoch', type=int, default=1)
    return parser.parse_args()


def test_td3(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    # train_envs = gym.make(args.task)
    if args.training_num > 1:
        train_envs = SubprocVectorEnv(
            [lambda: gym.make(args.task) for _ in range(args.training_num)])
    else:
        train_envs = gym.make(args.task)
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    # th??m actor
    actor1 = Actor(
        net_a, args.action_shape, max_action=args.max_action,
        device=args.device).to(args.device)
    actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)
    actor2 = Actor(
        net_a, args.action_shape, max_action=args.max_action,
        device=args.device).to(args.device)
    actor2_optim = torch.optim.Adam(actor2.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.state_shape, args.action_shape,
                 hidden_sizes=args.hidden_sizes,
                 concat=True, device=args.device)
    net_c2 = Net(args.state_shape, args.action_shape,
                 hidden_sizes=args.hidden_sizes,
                 concat=True, device=args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    es = sepCEM(get_size(actor1), mu_init=get_params(actor1), 
                sigma_init=args.sigma_init, damp=args.damp, 
                damp_limit=args.damp_limit, pop_size=args.pop_size, 
                antithetic=not args.pop_size % 2, parents=args.pop_size // 2, 
                elitism=args.elitism)

    # if args.auto_alpha:
    #     target_entropy = -np.prod(env.action_space.shape)
    #     log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
    #     alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    #     args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = P3STD3Policy(
        actor1, actor1_optim, actor2, actor2_optim, critic1, critic1_optim, critic2, critic2_optim,
        tau=args.tau, gamma=args.gamma, exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        estimation_step=args.n_step, action_space=env.action_space, q_weight=args.q_weight, regularization_weight=args.regularization_weight)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_td3'
    log_path = os.path.join(args.logdir, args.task, 'td3', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    if not args.watch:
        # trainer
        result = trainer_v3(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, args.test_num,
            args.batch_size, save_fn=save_fn, logger=logger,
            update_per_step=args.update_per_step, test_in_train=False,
            pop_size=args.pop_size, n_grad=args.n_grad, actor_lr=args.actor_lr,
            max_step=args.max_step, start_step=args.start_step, es=es, 
            log_path=log_path, episode_per_epoch=args.episode_per_epoch, action_shape = args.action_shape)
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_td3()
