from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

from utils import softmax

class Bandit(object):
    """Bandit used in [1] as an example.

    [1] N. D. Daw, "Trial-by-trial data analysis using computational models,"
    Decision making, affect, and learning: Attention and performance XXIII,
    vol. 23, p. 1, 2011.
    """
    def __init__(self):
        self.n = 2

    def reward(self, action):
        """Return reward given the action.

            Action 0 has probability 0.8 of winning 1.
            Action 1 has probability 0.2 of winning 0.
        """
        probabilities = (0.8, 0.2)
        p = probabilities[action]
        if action >=0 and action < self.n:
            if np.random.rand() < p:
                r = 1
            else:
                r = 0
        else:
            print('Error: action out of range')
            r = None
        return r

class Agent(object):
    """Agent used in [1] as an example.

    [1] N. D. Daw, "Trial-by-trial data analysis using computational models,"
    Decision making, affect, and learning: Attention and performance XXIII,
    vol. 23, p. 1, 2011.
    """

    def __init__(self, bandit, alpha=0.25, beta=1):
        self.bandit = bandit
        self.Q = np.zeros(2)
        self.alpha = alpha
        self.beta = beta
        self.log = {'reward':[], 'action':[], 'Q(0)':[], 'Q(1)':[]}

    def run(self):
        p = softmax(self.Q, self.beta)
        actions = (0, 1)
        action = np.random.choice(actions, p=p)
        reward = self.bandit.reward(action)
        self.Q[action] += self.alpha * (reward - self.Q[action])
        
        self.log['reward'].append(reward)
        self.log['action'].append(action)
        self.log['Q(0)'].append(self.Q[0])
        self.log['Q(1)'].append(self.Q[1])

    def get_df(self):
        columns = ['action', 'reward', 'Q(0)', 'Q(1)']
        df = pd.DataFrame(self.log, columns=columns)
        return df

class BanditCard(object):
    def __init__(self):
        self.n = 4

    def reward(self, action):
        """Return reward given the action.

        The bandit has no cues.
        """
        actions_bet = (3, 8, 14, 23)
        p_win = 0.8
        if action >=0 and action < self.n:
            if np.random.rand() < p_win:
                r = actions_bet[action]
            else:
                r = -actions_bet[action]
        else:
            print('Error: action out of range')
            r = None
        return r

class AgentCard(object):
    def __init__(self, bandit, alpha=0.25, beta=1):
        self.bandit = bandit
        self.Q = np.zeros(4)
        self.alpha = alpha
        self.beta = beta
        self.log = {'reward':[], 'action':[], 'Q(0)':[], 'Q(1)':[],
                    'Q(2)':[], 'Q(3)':[]}

    def run(self):
        p = softmax(self.Q, self.beta)
        actions = (0, 1, 2, 3)
        action = np.random.choice(actions, p=p)
        reward = self.bandit.reward(action)
        self.Q[action] += self.alpha * (reward - self.Q[action])
        
        self.log['reward'].append(reward)
        self.log['action'].append(action)
        self.log['Q(0)'].append(self.Q[0])
        self.log['Q(1)'].append(self.Q[1])
        self.log['Q(2)'].append(self.Q[2])
        self.log['Q(3)'].append(self.Q[3])

    def get_df(self):
        columns = ['action', 'reward', 'Q(0)', 'Q(1)', 'Q(2)', 'Q(3)']
        df = pd.DataFrame(self.log, columns=columns)
        return df


class BanditCardCues(object):
    def __init__(self, n_cues=2, probs=(0.8, 0.2)):
        self.actions_bet = (3, 8, 14, 23)
        self.actions = range(len(self.actions_bet))
        self.n_actions = len(self.actions_bet)
        self.n_cues = n_cues
        self.cues = range(n_cues)
        self.probs = probs

    def get_cue(self):
        self.cue = np.random.choice(self.cues)
        return self.cue
        
    def reward(self, action):
        """Return reward given the action.

        The bandit has no cues.
        """
        p_win = self.probs[self.cue]
        if action in self.actions:
            if np.random.rand() < p_win:
                r = self.actions_bet[action]
            else:
                r = -self.actions_bet[action]
        else:
            print('Error: action out of range')
            r = None
        return r

class AgentCardCues(object):
    def __init__(self, bandit, alpha=0.05, beta=1):
        self.bandit = bandit
        # Q-values are stored as Q[cue, action]
        self.Q = np.zeros((bandit.n_cues, bandit.n_actions))
        self.alpha = alpha
        self.beta = beta
        self.log = defaultdict(list)

    def run(self):
        actions = self.bandit.actions
        cues = self.bandit.cues
        cue = self.bandit.get_cue()
        p = softmax(self.Q[cue,:], self.beta)
        action = np.random.choice(actions, p=p)
        reward = self.bandit.reward(action)
        self.Q[cue, action] += self.alpha * (reward - self.Q[cue, action])
        
        self.log['reward'].append(reward)
        self.log['action'].append(action)
        self.log['cue'].append(cue)
        for cue, action in product(cues, actions):
            key = 'Q({:d},{:d})'.format(cue, action)
            self.log[key].append(self.Q[cue, action])

    def get_df(self):
        columns = ['cue', 'action', 'reward']
        for cue, action in product(self.bandit.cues, self.bandit.actions):
            columns.append('Q({:d},{:d})'.format(cue, action))
        df = pd.DataFrame(self.log, columns=columns)
        return df
        

if __name__ == '__main__':
    bandit = BanditCardCues()
    agent = AgentCardCues(bandit, alpha=0.05)
    trials = 300
    for _ in range(trials):
        agent.run()

    df = agent.get_df()
    df.to_pickle('df/agent_cues.pkl')
