class PPO:
	def__init__(self, state_dim,action_dim,lrate_actor,lrate_critic, gamma, epochs, epsilon, has_continuous_action_space)
		self.gamma = gamma
		self.epsilon = epsilon
		self.epochs = epochs

	def select_action(self,state)

	self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

	def update(self,)
	#estimated retuns using Monte Carlo
	rewards = [];
	discounted_reward = 0; #initially
	discounted_reward = reward + (self.gamma*discounted_reward)
	#where is reward  
	rewards.insert(0,discounted_reward)

	#normalizing the rewards

	#Optimizing for K epochs:
	for _ in range(self.epochs):
		#evaluate old actions & values

		rtheta = #

		#finding the surrrogate loss
		advantage = rewards - state.value.detach()
		#surrogate loss original
		surr1 = ratios*advantages
		#surrogate loss clipped
		surr2 = torch.clamp(ratios, 1-self.epsilon,1+self.epsilon)*advantage

		#choose the right loss according to clipped surrogate function
		loss = -torch.min(surr1,surr2) + # 0.5*self(MSeLoss(state_values, rewards)) -.01*dist_entropy

		# take gradient step
            self.optimizer.zero_grad() #zero out the gradients since PyTorch accumulates the gradients
            loss.mean().backward() #use backward for computational efficiency in taking gradients

