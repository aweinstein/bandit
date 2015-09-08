from itertools import product
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import DataFrame
import pandas as pd
from joblib import Parallel, delayed

pd.options.display.float_format = '{:.2f}'.format
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
    def __init__(self, bandit, epsilon=None, tau=None, Q_init=None, alpha=None,
                 verbose=False):
        self.epsilon = epsilon
        self.tau = tau
        self.bandit = bandit
        self.actions = self.bandit.actions
        self.contexts = self.bandit.get_context_list()
        self.n = bandit.n
        self.Q_init = Q_init
        self.alpha = alpha
        self.verbose = verbose
        self.reset()
        
    def run(self):
        context = self.bandit.get_context()
        action = self.choose_action(context)
        reward = self.bandit.reward(self.actions[action])
        
        # Update action-value
        self.update_action_value(context, action, reward)
 
        # Keep track of performance
        self.log['context'].append(context)
        self.log['reward'].append(reward)
        self.log['action'].append(self.actions[action])
        self.log['Q(c,23)'].append(self.Q[context][0])
        self.log['Q(c,14)'].append(self.Q[context][1])
        self.log['Q(c,8)'].append(self.Q[context][2])
        self.log['Q(c,3)'].append(self.Q[context][3])

    def choose_action_greedy(self, context):  
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.bandit.n)
        else:
            action = np.argmax(self.Q[context])
        return action

    def choose_action_softmax(self, context):
        p = softmax(self.Q[context], self.tau)
        actions = range(self.n)
        action = np.random.choice(actions, p=p)
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
            if self.verbose:
                print('Using update rule with alpha {:.2f}.'.format(
                    self.alpha))
        else:
            self.update_action_value = self.update_action_value_sample_average
            if self.verbose:
                print('Using sample average update rule.')

        if self.epsilon is not None:
            self.choose_action = self.choose_action_greedy
            if self.verbose:
                print('Using epsilon-greedy with epsilon ' 
                      '{:.2f}.'.format(self.epsilon))
        elif self.tau:
            self.choose_action = self.choose_action_softmax
            if self.verbose:
                print('Using softmax with tau {:.2f}'.format(self.tau))
        else:
            print('Error: epsilon or tau must be set')
            sys.exit(-1)
            
        self.log = {'context':[], 'reward':[], 'action':[], 
                    'Q(c,23)':[], 'Q(c,14)':[], 'Q(c,8)':[], 'Q(c,3)': []}


def softmax(Qs, tau):
    """Compute softmax probabilities for all actions."""
    num = np.exp(Qs / tau)
    den = np.exp(Qs / tau).sum()
    return num / den

def sanity_check():
    """Check that the interaction and bookkeeping is OK.

    Set the agent to epsilon equal to 0.99. This makes 
    almost all the actions to be selected uniformly at random.
    The action value for each context should follow the expected
    reward for each context.
    """
    print('Running a contextual bandit experiment')
    cb = ContextualBandit()
    ca = ContextualAgent(cb, epsilon=0.99)
    steps = 10000
    for _ in range(steps):
        ca.run()
    rewards = np.array(cb.actions)
    df = DataFrame(ca.log, columns=('context', 'action', 'reward', 'Q(c,a)'))
    fn = 'sanity_check.csv'
    df.to_csv(fn, index=False)
    print('Sequence written in', fn)
    print()
    for context, prob in cb.contexts.items():
        print(context, ': ')
        print('samp : ', ca.Q[context])
        print(' teo : ', prob * rewards - (1 - prob) * rewards)
        print()
    globals().update(locals())

def run_single_softmax_experiment(tau, alpha):
    """Run experiment with agent using softmax update rule."""
    print('Running a contextual bandit experiment')
    cb = ContextualBandit()
    ca = ContextualAgent(cb, tau=tau, alpha=alpha)
    steps = 300
    
    for _ in range(steps):
        ca.run()
    df = DataFrame(ca.log, columns=('context', 'action', 'reward', 'Q(c,23)', 
                                    'Q(c,14)', 'Q(c,8)', 'Q(c,3)'))
    fn = 'softmax_experiment.csv'
    #df.to_csv(fn, index=False)
    #print('Sequence written in', fn)
    print(df)
    print(df[df['context']=='reward'].tail(10))
    print(df[df['context']=='neutral'].tail(10))
    print(df[df['context']=='punishment'].tail(10))
    globals().update(locals())

def softmax_trial(tau, alpha):
    cb = ContextualBandit()
    trials = 100
    steps = 300
    reward_trials = np.zeros(trials)
    for i in range(trials):
        ca = ContextualAgent(cb, tau=tau, alpha=alpha)
        for _ in range(steps):
            ca.run()
        reward_trials[i] = sum(ca.log['reward'])
    return (tau, alpha, reward_trials.mean(), reward_trials.std())
    
def run_grid_search_softmax_exp():
    """Grid search of optimal tau and alpha."""
    print('Running a contextual bandit experiment')
    taus = [0.1, 1, 2, 3, 5]
    alphas = [0.01, 0.05, 0.1, 0.2]
    res = Parallel(n_jobs=-1, verbose=10)(delayed(softmax_trial)
                                         (tau, alpha) for (tau, alpha) in
                                         product(taus, alphas))
    ps = ('tau', 'alpha', 'tot_reward_mean', 'tot_reward_std')
    df = DataFrame(dict([(k, [r[i] for r in res]) for i,k in enumerate(ps)]),
                    columns=ps)
    print(df)
    print()
    print('Best:')
    print(df.loc[df['tot_reward_mean'].idxmax()])
    globals().update(locals())
    
if __name__ == '__main__':
    run_grid_search_softmax_exp()
