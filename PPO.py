




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
	def __init__(self, 
				state_dim, 
				action_dim, 
				lrate_actor,
				lrate_critic, 
				gamma, 
				epochs, 
				epsilon, 
				has_continuous_action_space):
	"""
	Initializer for the 
	"""



	def select_action(self,):


	def update(self,):
	#estimated retuns using Monte Carlo	
