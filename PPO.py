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
				device,
				using_TD = False,
				action_std_init=1):

		self.discount_factor = gamma
		self.device = device
		self.clip_factor = epsilon
		self.epochs = epochs
		self.buffer = RolloutBuffer()
		self.action_dim = action_dim
		self.actor = actor.to(device)
		self.critic = critic.to(device)
		self.optimizer = torch.optim.Adam(
			[ 
				{'params': self.actor.parameters(), 'lr': lrate_actor},
				{'params': self.critic.parameters(), 'lr': lrate_critic}
			]
		)
		self.using_TD = using_TD

		self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
		
		self.continuous = has_continuous_action_space

	def select_action(self, state):

		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			print(state.shape)
			if self.continuous:
				action_mean = self.actor(state)

				cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device).type('torch.FloatTensor')
				dist = MultivariateNormal(action_mean, cov_mat)
			else:
				action_probs = self.actor(state)
				dist = Categorical(action_probs)

		action = dist.sample()
		action_logprob = dist.log_prob(action)

		self.buffer.actions.append(action.detach())
		self.buffer.states.append(state)
		self.buffer.logprobs.append(action_logprob.detach())

		if self.continuous:
			return action.detach().cpu().numpy().flatten()

		return action.detach().item()


	def evaluate(self, state, action):

		if self.continuous:
			action_mean = self.actor(state)
			action_var = self.action_var.expand_as(action_mean)
			cov_mat = torch.diag_embed(action_var).to(self.device).type('torch.FloatTensor')
			dist = MultivariateNormal(action_mean, cov_mat)
			
			# for single action continuous environments
			if self.action_dim == 1:
				action = action.reshape(-1, self.action_dim)

		else:
			action_probs = self.actor(state)
			dist = Categorical(action_probs)

		action_logprobs = dist.log_prob(action)
		state_values = self.critic(state)
		dist_entropy = dist.entropy()

		return action_logprobs, state_values, dist_entropy

	def monte_carlo(self):

		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.discount_factor * discounted_reward)

			rewards.insert(0, discounted_reward)
			
		# Normalizing the rewards
		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		return rewards

	def TD_get_advantage(self, rewards, values):

		advantages, gae = [], 0

		for i in reversed(range(len(rewards))):

			# TD error
			next_value = 0 if i + 1 == len(rewards) else values[i + 1]
			delta = rewards[i] + self.discount_factor * next_value - values[i]
			gae = delta + self.discount_factor * 0.95 * gae
			advantages.insert(0, delta)

		return torch.tensor(advantages, dtype=torch.float32).to(self.device)

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
			logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
			state_values = torch.squeeze(state_values)


			if not self.using_TD:
				
				#finding the surrrogate loss
				advantage = rewards - state_values.detach() 
			else: 
				advantage = self.TD_get_advantage(self.buffer.rewards, state_values.detach())




			ratios = torch.exp(logprobs - old_logprobs)
            # Utilizing the comb
			
			#surrogate loss original
			surr1 = ratios * advantage
			#surrogate loss clipped
			surr2 = torch.clamp(ratios, 1 - self.clip_factor,1 + self.clip_factor) * advantage

			loss = -torch.min(surr1, surr2) +  0.5 * F.mse_loss(state_values, rewards) - 0.01 * dist_entropy

			# take gradient step
			self.optimizer.zero_grad() #zero out the gradients since PyTorch accumulates the gradients
			loss.mean().backward() #use backward for computational efficiency in taking gradients
			self.optimizer.step()

		self.buffer.clear()


	def save_actor(self, path):
		torch.save(self.actor.state_dict(), path) # made an edit here -terri
    
	def load_actor(self, path):
		self.actor.load_state_dict(torch.load(path, self.device)) # made an edit here - terri


