from collections import defaultdict
from itertools import product
from types import MethodType

import numpy as np
import pandas as pd

from utils import softmax

class Bandit(object):
    """Bandit used in [1] as an example.

    The bandit has two levers. Reward is 0 or 1. Lever 0 has a probability of
    winning of 0.8, and lever 1 a probability of winning of 0.2.

    The bandit behavior can be generalized by passing a function that compute
    the reward given the action and the trial number.

    [1] N. D. Daw, "Trial-by-trial data analysis using computational models,"
    Decision making, affect, and learning: Attention and performance XXIII,
    vol. 23, p. 1, 2011.

    """
    def __init__(self, reward_func=None):
        """Set `reward_func` to a function with parameters (self, action, trial)
        to define a new reward structure.
        """
        self.n = 2
        self.trial = 0
        if reward_func is not None:
            self.compute_reward = MethodType(reward_func, self)

    def reward(self, action):
        """Return reward given the action."""
        self.trial += 1
        r = self.compute_reward(action, self.trial)
        return r

    def compute_reward(self, action, trial):
        """Compute the reward.
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

    def __init__(self, bandit, alpha=0.25, beta=1, model='value'):
        self.bandit = bandit
        self.alpha = alpha
        self.beta = beta
        self.Q = np.zeros(2)
        self.pi = np.zeros(2)
        self.log = defaultdict(list)
        self.model = model
        if model not in ('value', 'policy'):
            raise ValueError("`model` must be one of `value`, `policy`, "
                             "got %r" % model)
        if model == 'value':
            self.update = self.update_value
            self.choose_action = self.choose_action_value
            print(f'Value update agent with alpha={alpha} and beta={beta}')

        else:
            #self.update = self.update_policy
            self.update = self.update_policy_daw
            self.choose_action = self.choose_action_policy
            self.reward_hat = 0.1 * np.ones(4)
            print(f'Policy update agent with alpha={alpha} and beta={beta}')

    def run(self):
        p = self.choose_action()
        actions = (0, 1)
        action = np.random.choice(actions, p=p)
        reward = self.bandit.reward(action)
        self.update(action, reward)

        # log the results
        self.log['reward'].append(reward)
        self.log['action'].append(action)
        if self.model == 'value':
            self.log['Q(0)'].append(self.Q[0])
            self.log['Q(1)'].append(self.Q[1])
        elif self.model == 'policy':
            self.log['pi(0)'].append(self.pi[0])
            self.log['pi(1)'].append(self.pi[1])
            self.log['r_hat'].append(self.reward_hat.mean())

    def choose_action_value(self):
        """Compute actions probabilities for value learning."""
        return softmax(self.Q, self.beta)

    def choose_action_policy(self):
        """Compute actions probabilities for policy learning."""
        return softmax(self.pi, self.beta)

    def update_value(self, action, reward):
        """Value model update rule.

        See Eq. (2) of [1].
        """
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def update_policy_daw(self, action, reward):
        """Value model update rule.

        See Eq. (12) of [1].
        """
        self.reward_hat = np.roll(self.reward_hat, -1)
        self.reward_hat[-1] = reward
        # self.pi[action] += 0.1 * (reward - self.reward_hat.mean())
        self.pi[action] += self.alpha * (reward - 0.5)

    def update_policy(self, action, reward):
        """Value model update rule.

        Using Dayan and Abbot or Sutton and Barto formulation.
        """
        self.reward_hat = np.roll(self.reward_hat, -1)
        self.reward_hat[-1] = reward
        #r_bar = self.reward_hat.mean()
        r_bar = 0.5
        probs = softmax(self.pi, self.beta)
        for a in (0,1):  # (0, 1) should be something like self.actions
            indicator = 1 if a == action else 0
            self.pi[a] += self.alpha * (reward - r_bar) * (indicator - probs[a])

    def get_df(self):
        if self.model == 'value':
            columns = ['action', 'reward', 'Q(0)', 'Q(1)']
        else:
            columns = ['action', 'reward', 'pi(0)', 'pi(1)', 'r_hat']
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
    def __init__(self, n_cues=3, probs=(0.8, 0.2, 0.5)):
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
        """Return reward given the action and current cue."""
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


def bandit_card_cues_experiment():
    bandit = BanditCardCues()
    agent = AgentCardCues(bandit, alpha=0.05)
    trials = 300
    for _ in range(trials):
        agent.run()

    df = agent.get_df()
    df.to_pickle('df/agent_cues.pkl')

def simple_bandit_experiment():
    #np.random.seed(42)
    bandit =Bandit()
    agent = Agent(bandit, model='policy', beta=0.2)
    trials = 300
    for _ in range(trials):
        agent.run()
    df = agent.get_df()
    import vis
    import matplotlib.pyplot as plt
    plt.close('all')
    vis.plot_simple_bandit(df)
    return df


def bee_experiment():
    def bee_reward(self, action, trial):
        if trial <= 100:
            rewards = (2, 4)
        else:
            rewards = (4, 2)
        return rewards[action] * np.random.choice(2)

    #np.random.seed(42)
    bandit = Bandit(bee_reward)
    #agent = Agent(bandit, model='value', alpha=0.1, beta=1.)
    agent = Agent(bandit, model='policy', alpha=0.4, beta=0.2)
    trials = 200
    for _ in range(trials):
        agent.run()
    df = agent.get_df()

    #compute sum visits
    actions = df['action']
    trials = len(df)
    sv_0, sv_1 = np.zeros(trials), np.zeros(trials)
    if actions[0] == 0:
        sv_0[0] = 1
    else:
        sv_1[0] = 1
    for i, a in actions[1:].iteritems():
        if a == 0:
            sv_0[i] = sv_0[i-1] + 1
            sv_1[i] = sv_1[i-1]
        else:
            sv_1[i] = sv_1[i-1] + 1
            sv_0[i] = sv_0[i-1]

    # visualisation
    import matplotlib.pyplot as plt
    import vis
    plt.close('all')
    plt.figure()
    plt.plot(sv_0)
    plt.plot(sv_1)
    vis.plot_simple_bandit(agent.get_df())
    plt.show()

    return df

if __name__ == '__main__':
    # print('Running the bees experiment')
    # df = bee_experiment()
    print('Running the simple bandit experiment')
    df = simple_bandit_experiment()
