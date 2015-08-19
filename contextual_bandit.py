import bisect
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2

class ContextualBandit(object):
    def __init__(self):
        # Contexts and their probabilities of wining
        self.contexts = {'punishment': 0.2,
                         'neutral': 0.5,
                         'reward': 0.8}
        self.actions = (3, 8, 17, 23)
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
        # self.rewards_seq.append(reward)
        # self.actions.append(action)
        # self.k_reward += 1
        # correct = (action == self.bandit.optimal_action()) * 100
        # self.optimal_actions.append(correct)

    def choose_action_greedy(self, context):  
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.bandit.n)
        else:
            action = np.argmax(self.Q[context])
        return action

    def choose_action_softmax(self, context):
        p = softmax(self.Q[context], self.tau)
        actions = range(self.n)
        action = choose(actions, p)
        return action
        
    def update_action_value_sample_average(self, context, action, reward):
        k = self.k_actions[context, action]
        self.Q[context][action] +=  (1 / k) * (reward - self.Q[action])
        self.k_actions[context, action] += 1

    def update_action_value_constant_alpha(self, context, action, reward):
        error = reward - self.Q[context][action]
        self.Q[context][action] += self.alpha * error

    def reset(self):
        self.Q = {}
        for context in self.contexts:
            if self.Q_init:
                self.Q[context] = self.Q_init * np.ones(self.n)
            else: # init with small random numbers to avoid ties
                self.Q[context] = np.random.uniform(0, 1e-4, self.n)

        if self.alpha:
            self.update_action_value = self.update_action_value_constant_alpha
            print('Using update rule with alpha {:.2f}.'.format(self.alpha))
        else:
            self.update_action_value = self.update_action_value_sample_average
            print('Using sample average update rule.')

        if self.epsilon is not None:
            self.choose_action = self.choose_action_greedy
            print('Using epsilon-greedy.')
        elif self.tau:
            self.choose_action = self.choose_action_softmax
            print('Using softmax.')
        else:
            print('Error: epsilon or tau must be set')
            sys.exit(-1)

        # number of steps for each action
        self.k_actions = np.ones((len(self.contexts), self.n))

        # self.rewards = []
        # self.rewards_seq = []
        # self.actions = []
        # self.k_reward = 1
        # self.average_reward = 0
        # self.optimal_actions = []
        
def softmax(Qs, tau):
    """Compute softmax probabilities for all actions."""
    num = np.exp(Qs / tau)
    den = np.exp(Qs / tau).sum()
    return num / den

# TODO: Replace by np.random.choice with parameter `p`
def choose(a, p):
    """Choose randomly an item from `a` with pmf given by p.
    a : list
    p : probability mass function
    """
    intervals = [sum(p[:i]) for i in range(len(p))]
    item = a[bisect.bisect(intervals, np.random.rand()) - 1]
    return item


if __name__ == '__main__':
    print('Running a contextual bandit experiment')
    cb = ContextualBandit()
    ca = ContextualAgent(cb, tau=0.1, alpha=0.1)
    for _ in range(10):
        ca.run()
    
