class PPO:
	def__init__(self, state_dim,action_dim,lrate_actor,lrate_critic, gamma, epochs, epsilon, cont_action_space)
		self.gamma = gamma
		self.epsilon = epsilon
		self.epochs = epochs
		self.buffer = RolloutBuffer()

		self.policy = ActorCritic(state_dim,action_dim,cont_action_space,action_std,init).to(device)
		self.optimizer = torch.optim.Adam([{'params':self.policy.actor.parameters(),'lr':lrate_actor},
			{'params': self.policy.critic.parameters(),'lr':lrate_critic}])
		
        self.policy_old = ActorCritic(state_dim, action_dim, cont_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

	def select_action(self,state):

		self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

	def update(self):
		#estimated retuns using Monte Carlo
		rewards = [];
		discounted_reward = 0; #initially
		discounted_reward = reward + (self.gamma*discounted_reward)
		#where is reward  
		rewards.insert(0,discounted_reward)

		#normalizing the rewards
		rewards = torch.tensor(rewards, dtype=torch.float3).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
		#Optimizing for K epochs:
		for _ in range(self.epochs):

			#evaluate old actions & values
	        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
			rtheta = ratios = torch.exp(logprobs - old_logprobs.detach())

			#finding the surrrogate loss
			advantage = rewards - state_values.detach()
			#surrogate loss original
			surr1 = ratios*advantages
			#surrogate loss clipped
			surr2 = torch.clamp(ratios, 1-self.epsilon,1+self.epsilon)*advantage

			#choose the right loss according to clipped surrogate function
			loss = -torch.min(surr1,surr2) + # 0.5*self(MSeLoss(state_values, rewards)) -.01*dist_entropy

			# take gradient step
	            self.optimizer.zero_grad() #zero out the gradients since PyTorch accumulates the gradients
	            loss.mean().backward() #use backward for computational efficiency in taking gradients
	            self.optimizer.step()

	        #assigning new weights into old policy
	        self.policy_old.load_state_dict(self.policy.state_dict())

	        self.buffer.clear()
