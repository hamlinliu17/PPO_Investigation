import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO:
	"""
	we dont need the actor critic
	 we just need to supply 2 NN modules, one is the actor one is the critic

	Update:
		we just need to get returns


	MC_function:

	TD_function: Do a little more reading
				what we need for this part


	
	"""
	def __init__(self,
				actor: nn.Module,
				critic: nn.Module,
				action_dim,
				lrate_actor,
				lrate_critic, 
				gamma, 
				epochs, 
				epsilon,
				has_continuous_action_space,
				action_std_init=1,
				device):

		self.discount_factor = gamma

		self.clip_factor = epsilon
		self.epochs = epochs
		self.buffer = RolloutBuffer()
		self.action_dim = action_dim
		self.actor = actor.to(device) 
		self.critic = critic.to(device)
		self.optimizer = torch.optim.Adam(
			[ 
				{'params': self.policy.actor.parameters(), 'lr': lrate_actor},
				{'params': self.policy.critic.parameters(), 'lr': lrate_critic}
			]
		)

		self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
		
		self.continuous = has_continuous_action_space

	def select_action(self, state):

		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			if self.continuous:
				action_mean = self.actor(state)
				cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
				dist = MultivariateNormal(action_mean, cov_mat)
			else:
				action_probs = self.actor(state)
				dist = Categorical(action_probs)

		action = dist.sample()
		action_logprob = dist.log_prob(action)

		self.buffer.actions.append(action.detach())
		self.buffer.state.append(state)
		self.buffer.logprobs.append(action_logprob.detach())

		if self.continuous:
			return action.detach().cpu().numpy().flatten()

		return action.detach().item()


	def evaluate(self, state, action):

		if self.has_continuous_action_space:
			action_mean = self.actor(state)
			action_var = self.action_var.expand_as(action_mean)
			cov_mat = torch.diag_embed(action_var).to(self.device)
			dist = MultivariateNormal(action_mean, cov_mat)
			
			# for single action continuous environments
			if self.action_dim == 1:
				action = action.reshape(-1, self.action_dim)

		else:
			action_probs = self.actor(state)
			dist = Categorical(action_probs)

		action_logprobs = dist.log_prob(action)
		state_values = self.critic(state)

		return action_logprobs, state_values

	def monte_carlo(self):

		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
			
		# Normalizing the rewards
		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		return rewards


	def update(self):
		# TODO: finish this up and test it around
		rewards = self.monte_carlo()

		#normalizing the rewards

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
		old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
		old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

		#Optimizing for K epochs:
		for _ in range(self.epochs):
			#evaluate old actions & values
			logprobs, state_values = self.evaluate(old_states, old_actions)

			ratios = torch.exp(logprobs - old_logprobs)

			#finding the surrrogate loss
			advantage = rewards - state_values.detach()
			#surrogate loss original
			surr1 = ratios * advantage
			#surrogate loss clipped
			surr2 = torch.clamp(ratios, 1 - self.clip_factor,1 + self.clip_factor) * advantage

			# Utilizing the comb

			loss = -torch.min(surr1, surr2) +  0.5 * F.MSELoss(state_values, rewards)

			# take gradient step
			self.optimizer.zero_grad() #zero out the gradients since PyTorch accumulates the gradients
			loss.mean().backward() #use backward for computational efficiency in taking gradients

		self.buffer.clear()


	def save_actor(self, path):
		torch.save(self.actor.state_dict, path)

	def load_actor(self, path, actor_test: nn.Module):
		self.actor = actor_test.load_state_dict(torch.load(path, device=self.device))


