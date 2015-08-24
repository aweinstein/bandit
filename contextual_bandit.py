from collections import defaultdict
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2

class ContextualBandit(object):
    def __init__(self):
        # Contexts and their probabilities of winning
        self.contexts = {'punishment': 0.2,
                         'neutral': 0.5,
                         'reward': 0.8}
        self.actions = (23, 14, 8, 3)
        self.n = len(self.actions)
        self.get_context()

    def get_context_list(self):
        return list(self.contexts.keys())
        
    def get_context(self):
        # Note: Nothing prevents the agent to call this functions several
        # times, without calling self.reward in between. Potentially this
        # can be used by the agent to cheat.
        self.context = np.random.choice(list(self.contexts.keys()))
        return self.context

    def reward(self, action):
        if action not in self.actions:
            print('Error: action not in', self.actions)
            sys.exit(-1)
        p = self.contexts[self.context]
        if np.random.rand() < p:
            r = action
        else:
            r = -action
        return r
        
class ContextualAgent(object):
    def __init__(self, bandit, epsilon=None, tau=None, Q_init=None, alpha=None):
        self.epsilon = epsilon
        self.tau = tau
        self.bandit = bandit
        self.actions = self.bandit.actions
        self.contexts = self.bandit.get_context_list()
        self.n = bandit.n
        self.Q_init = Q_init
        self.alpha = alpha
        self.reset()
        
    def run(self):
        context = self.bandit.get_context()
        action = self.choose_action(context)
        reward = self.bandit.reward(self.actions[action])
        
        # Update action-value
        self.update_action_value(context, action, reward)
 
        # Keep track of performance
        self.rewards_seq[context].append(reward)
        self.actions_seq[context].append(action)

    def choose_action_greedy(self, context):  
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.bandit.n)
        else:
            action = np.argmax(self.Q[context])
        return action

    def choose_action_softmax(self, context):
        p = softmax(self.Q[context], self.tau)
        actions = range(self.n)
        action = np.random.choise(actions, p=p)
        return action
        
    def update_action_value_sample_average(self, context, action, reward):
        k = self.k_actions[context][action]
        self.Q[context][action] +=  ((1 / k) *
                                     (reward - self.Q[context][action]))
        self.k_actions[context][action] += 1

    def update_action_value_constant_alpha(self, context, action, reward):
        error = reward - self.Q[context][action]
        self.Q[context][action] += self.alpha * error

    def reset(self):
        self.Q = {}
        self.k_actions = {}
        for context in self.contexts:
            if self.Q_init:
                self.Q[context] = self.Q_init * np.ones(self.n)
            else: # init with small random numbers to avoid ties
                self.Q[context] = np.random.uniform(0, 1e-4, self.n)
                    # number of steps for each action
            self.k_actions[context] = np.ones(self.n)

        if self.alpha:
            self.update_action_value = self.update_action_value_constant_alpha
            print('Using update rule with alpha {:.2f}.'.format(self.alpha))
        else:
            self.update_action_value = self.update_action_value_sample_average
            print('Using sample average update rule.')

        if self.epsilon is not None:
            self.choose_action = self.choose_action_greedy
            print('Using epsilon-greedy with epsilon ' 
                  '{:.2f}.'.format(self.epsilon))
        elif self.tau:
            self.choose_action = self.choose_action_softmax
            print('Using softmax.')
        else:
            print('Error: epsilon or tau must be set')
            sys.exit(-1)
            
        self.rewards_seq = defaultdict(list)
        self.actions_seq = defaultdict(list)


def softmax(Qs, tau):
    """Compute softmax probabilities for all actions."""
    num = np.exp(Qs / tau)
    den = np.exp(Qs / tau).sum()
    return num / den

def sanity_check():
    """Check that the interection and bookkeeping is OK.

    Set the agent to with epsilon equal to 0.99. This makes 
    almost all the actions to be selected uniformly at random.
    The action value for each context should follow the expected
    reward for each context.
    """
    print('Running a contextual bandit experiment')
    cb = ContextualBandit()
    #ca = ContextualAgent(cb, tau=0.1, alpha=0.1)
    ca = ContextualAgent(cb, epsilon=0.99)
    steps = 10000
    for _ in range(steps):
        ca.run()
    rewards = np.array(cb.actions)
    print()
    for context, prob in cb.contexts.items():
        print(context, ': ')
        print('samp : ', ca.Q[context])
        print(' teo : ', prob * rewards - (1 - prob) * rewards)
        print()

if __name__ == '__main__':
    sanity_check()
