from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
from esrl.policy.ddpg import DDPGPolicy
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def kl(mean1, std1, mean2, std2):
        distribution1   = Normal(mean1, std1)
        distribution2   = Normal(mean2, std2)

        return kl_divergence(distribution1, distribution2).float().to(device)


class P3STD3Policy(DDPGPolicy):
    """Implementation of TD3, arXiv:1802.09477.
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param float exploration_noise: the exploration noise, add to the action.
        Default to ``GaussianNoise(sigma=0.1)``
    :param float policy_noise: the noise used in updating policy network.
        Default to 0.2.
    :param int update_actor_freq: the update frequency of actor network.
        Default to 2.
    :param float noise_clip: the clipping range used in updating policy network.
        Default to 0.5.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor1: torch.nn.Module,
        actor1_optim: torch.optim.Optimizer,
        actor2: torch.nn.Module,
        actor2_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        q_weight: float = 0.1,
        regularization_weight: float = 0.005,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None, None, None, None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step, **kwargs
        )
        # t thêm vào 2 actor với q_weight, regularization_weight
        self.actor1, self.actor1_old = actor1, deepcopy(actor1)
        self.actor1_old.eval()
        self.actor1_optim = actor1_optim
        self.actor2, self.actor2_old = actor2, deepcopy(actor2)
        self.actor2_old.eval()
        self.actor2_optim = actor2_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._policy_noise = policy_noise
        self._freq = update_actor_freq
        self._noise_clip = noise_clip
        self._cnt = 0
        self._last = 0
        self.q_weight = q_weight
        self.regularization_weight = regularization_weight

    def train(self, mode: bool = True) -> "P3STD3Policy":
        self.training = mode
        self.actor1.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self
    
    def train2(self, mode: bool = True) -> "P3STD3Policy":
        #print("!!!!!")
        # cái này có chạy nha m
        self.training = mode
        self.actor1.train(mode)
        self.actor2.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)
        #update para của 2 actor 
        self.soft_update(self.actor1_old, self.actor1, self.tau)
        # self.soft_update(self.actor2_old, self.actor2, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        # cái này cũng có chạy
        batch = buffer[indices]  # batch.obs: s_{t+n}
        act1_ = self(batch, model="actor1_old", input="obs_next").act
        act2_ = self(batch, model="actor2_old", input="obs_next").act
        noise = torch.randn(size=act1_.shape, device=act1_.device) * self._policy_noise
        if self._noise_clip > 0.0:
            noise = noise.clamp(-self._noise_clip, self._noise_clip)
        act1_ += noise
        act2_ += noise
        target_q1 = torch.min(
            self.critic1_old(batch.obs_next, act1_),
            self.critic2_old(batch.obs_next, act1_),
        )
        target_q2 = torch.min(
            self.critic1_old(batch.obs_next, act2_),
            self.critic2_old(batch.obs_next, act2_),
        )
        #công thức tính target theo paper DARC
        target_q = self.regularization_weight * torch.min(target_q1, target_q2) + (1 - self.regularization_weight) * torch.max(target_q1, target_q2)
        return target_q

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor1",
        model2: str = "actor2",
        input: str = "obs",
        flag: bool = False,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.
        :return: A :class:`~tianshou.data.Batch` which has 2 keys:
            * ``act`` the action.
            * ``state`` the hidden state.
        .. seealso::
            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        
        #t đang bí ko bk code khúc này sao khi mà có 2 actor
        model = getattr(self, model)
        obs = batch[input]
        if flag == False:
            actions, hidden = model(obs, state=state, info=batch.info)
        else:
            model2 = getattr(self, model2)
            actions, hidden = model(obs, state=state, info=batch.info)
            actions2, hidden2 = model2(obs, state=state, info=batch.info)
            q1 = self.critic1(obs, actions)
            q2 = self.critic2(obs, actions2)

            if q2 > q1:
                actions, hidden = actions2, hidden2
        return Batch(act=actions, state=hidden)
    

    def learn(self,best_actor: Any,action_shape:Any, batch: Batch, beta, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim
        )
        td2, critic2_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer


        dist = self(batch).act
        best_dist = best_actor(batch).act
        std = torch.ones([1, action_shape[0]]).float().to(device)
        KL = kl(dist, std, best_dist, std)

        # actor
        if self._cnt % self._freq == 0:
            actor1_loss = (-self.critic1(batch.obs, self(batch,model="actor1", eps=0.0).act) + beta * KL).mean()
            self.actor1_optim.zero_grad()
            actor1_loss.backward()
            self._last1 = actor1_loss.item()
            self.actor1_optim.step()
            # update hàm loss của 2 actor
            # actor2_loss = (-self.critic2(batch.obs, self(batch,model="actor2", eps=0.0).act) + beta * KL).mean()
            # self.actor2_optim.zero_grad()
            # actor2_loss.backward()
            # self._last2 = actor2_loss.item()
            # self.actor2_optim.step()
            self.sync_weight()
        self._cnt += 1
        return {
            #ko biết sẽ return cái gì
            "loss/actor1": self._last1,
            #"loss/actor2": self._last2,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
