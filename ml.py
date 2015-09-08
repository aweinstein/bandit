"""Maximum Likelihood estimation of bandit parameters.

See [1] for details.

[1] N. D. Daw, "Trial-by-trial data analysis using computational models,"
Decision making, affect, and learning: Attention and performance XXIII,
vol. 23, p. 1, 2011.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Bandit(object):
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

class BanditCard(object):
    def __init__(self):
        self.n = 4

    def reward(self, action):
        """Return reward given the action."""
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
        
class Agent(object):
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

def neg_log_likelihood(alphabeta, data):
    alpha, beta = alphabeta
    actions, rewards = data['action'], data['reward']
    prob_log = 0
    Q = np.zeros(len(data.keys()) - 2) #np.zeros(2)
    for action, reward in zip(actions, rewards):
        Q[action] += alpha * (reward - Q[action])
        prob_log += np.log(softmax(Q, beta)[action])
    return -prob_log
    
def ml_estimation(log):
    r = minimize(neg_log_likelihood, [0.1,0.1], args=(log,),
                 method='Nelder-Mead')
    return r


def softmax(Qs, beta):
    """Compute softmax probabilities for all actions."""
    num = np.exp(Qs * beta)
    den = np.exp(Qs * beta).sum()
    return num / den

def plot_ml(log, alpha_hat, beta_hat):
    from itertools import product
    n = 50
    alphas = np.linspace(0, 1, n)
    betas = np.linspace(0, 2, n)
    Alpha, Beta = np.meshgrid(alphas, betas)
    Z = np.zeros(len(Alpha) * len(Beta))
    for i, (a, b) in enumerate(product(alphas, betas)):
        Z[i] = neg_log_likelihood((a, b), log)
    Z.resize((len(alphas), len(betas)))
    plt.contourf(Alpha, Beta, Z.T, 50)
    plt.plot(alpha_hat, beta_hat, 'r+', ms=10)
    plt.plot(alpha, beta, 'rs', ms=5)
    
def simple_bandit_experiment():
    b = Bandit()
    alpha = 0.1
    beta = 1.2
    print('alpha: {:.2f} beta: {:.2f}\n'.format(alpha, beta))
    agent = Agent(b, alpha, beta)
    trials = 1000

    for _ in range(trials):
        agent.run()
    df = pd.DataFrame(agent.log,
                      columns=('action', 'reward', 
                               'Q(0)', 'Q(1)'))
    r = ml_estimation(agent.log)
    alpha_hat, beta_hat = r.x
    print(r)
    
    plot_ml(agent.log, alpha_hat, beta_hat)

if __name__ == '__main__':
    b = BanditCard()
    alpha = 0.1
    beta = 0.7
    print('alpha: {:.2f} beta: {:.2f}\n'.format(alpha, beta))
    agent = AgentCard(b, alpha, beta)
    trials = 1000

    for _ in range(trials):
        agent.run()
    df = pd.DataFrame(agent.log,
                      columns=('action', 'reward', 
                               'Q(0)', 'Q(1)', 'Q(2)', 'Q(3)'))
    print('Total reward: {:d}\n'.format(df['reward'].sum()))
    r = ml_estimation(agent.log)
    alpha_hat, beta_hat = r.x
    print(r)

    plot_ml(agent.log, alpha_hat, beta_hat)

