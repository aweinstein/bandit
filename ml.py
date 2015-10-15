"""Maximum Likelihood estimation of bandit parameters.

See [1] for details.

[1] N. D. Daw, "Trial-by-trial data analysis using computational models,"
Decision making, affect, and learning: Attention and performance XXIII,
vol. 23, p. 1, 2011.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import save_figs_as_pdf

Data_Behavior_Dir = 'data_behavior'

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
    Q = np.zeros(len(data.keys()) - 2)
    for action, reward in zip(actions, rewards):
        Q[action] += alpha * (reward - Q[action])
        prob_log += np.log(softmax(Q, beta)[action])
    return -prob_log
    
def ml_estimation(log, method_name='Nelder-Mead'):
    #opt = {'maxfev': 800, 'maxiter':800}
    r = minimize(neg_log_likelihood, [0.1,0.1], args=(log,),
                 method=method_name,
                 #options=opt)
                 )
    return r

def fit_model(pkl):
    log = make_log(pkl)
    r = ml_estimation(log, 'Nelder-Mead')
    if r.status != 0:
        print('trying with Powell')
        r = ml_estimation(log, 'Powell')
    return r

def softmax(Qs, beta):
    """Compute softmax probabilities for all actions."""
    num = np.exp(Qs * beta)
    den = np.exp(Qs * beta).sum()
    return num / den

def plot_ml(ax, log, alpha, beta, alpha_hat, beta_hat):
    from itertools import product
    n = 50
    alpha_max = 0.2
    beta_max = 1.3
    if alpha is not None:
        alpha_max = alpha_max if alpha < alpha_max else 1.1 * alpha
        beta_max = beta_max if beta < beta_max else 1.1 * beta
    alphas = np.linspace(0, alpha_max, n)
    betas = np.linspace(0, beta_max, n)
    Alpha, Beta = np.meshgrid(alphas, betas)
    Z = np.zeros(len(Alpha) * len(Beta))
    for i, (a, b) in enumerate(product(alphas, betas)):
        Z[i] = neg_log_likelihood((a, b), log)
    Z.resize((len(alphas), len(betas)))
    ax.contourf(Alpha, Beta, Z.T, 50)
    if alpha is not None:
        ax.plot(alpha, beta, 'rs', ms=5)
    if alpha_hat is not None:
        ax.plot(alpha_hat, beta_hat, 'r+', ms=10)
    ax.set_xlabel(r'$\alpha$', fontsize=20)
    ax.set_ylabel(r'$\beta$', fontsize=20)
    return 

def simple_bandit_experiment():
    b = Bandit()
    alpha = 0.1
    beta = 1.2
    print('alpha: {:.2f} beta: {:.2f}\n'.format(alpha, beta))
    agent = Agent(b, alpha, beta)
    trials = 1000

    for _ in range(trials):
        agent.run()
    # df = pd.DataFrame(agent.log,
    #                   columns=('action', 'reward', 
    #                            'Q(0)', 'Q(1)'))
    r = ml_estimation(agent.log)
    alpha_hat, beta_hat = r.x
    print(r)
    fig, ax = plt.subplots(1, 1)
    plot_ml(ax, agent.log, alpha, beta, alpha_hat, beta_hat)
    plt.show()

def card_bandit_experiment():
    b = BanditCard()
    alpha = 0.2
    beta = 0.5
    print('alpha: {:.2f} beta: {:.2f}\n'.format(alpha, beta))
    agent = AgentCard(b, alpha, beta)
    trials = 120

    for _ in range(trials):
        agent.run()
    df = pd.DataFrame(agent.log,
                      columns=('action', 'reward', 
                               'Q(0)', 'Q(1)', 'Q(2)', 'Q(3)'))
    df.to_csv('data.csv', index_label='trial')
    print('Total reward: {:d}\n'.format(df['reward'].sum()))
    r = ml_estimation(agent.log)
    alpha_hat, beta_hat = r.x
    print(r)

    fig, ax = plt.subplots(1, 1)
    plot_ml(ax, agent.log, alpha, beta, alpha_hat, beta_hat)
    plt.show()
    globals().update(locals())

def make_log(pkl):
    """Make a log dictionary from behavioral data."""
    fn = os.path.join(Data_Behavior_Dir, pkl)
    df = pd.read_pickle(fn)
    cue = 1
    # This is ugly!!! Fixit !!!
    action = df[df['cues']==cue]['choices'].values
    reward = df[df['cues']==cue]['rewards'].values
    action = list(map(lambda x: {3:0, 8:1, 14:2, 23:3}[x], action))
    log = {'action':action, 'reward':reward, 'foo':None, 'bar':None,
           'foobar':None, 'barfoo':None}
    return log

def plot_single_subject(fn, ax, r):
    log = make_log(fn)
    if r.status == 0:
        alpha, beta = r.x
        plot_ml(ax, log, alpha, beta, None, None)
        title = 'Subject {:s}'.format(fn[:2])
    else:
        plot_ml(ax, log, None, None, None, None)
        title = 'Subject {:s}, not converged'.format(fn[:2])
    ax.set_title(title)

def fit_behavioral_data():
    """Fit a model for all subjects.

    The data has been previously parsed by pase.py.
    """
    pkls = os.listdir(Data_Behavior_Dir)
    pkls.sort()
    data = {'alpha':[], 'beta':[], 'subject': [], 'status':[]}
    figs = []
    for pkl in pkls:
        print(pkl)
        r = fit_model(pkl)
        alpha, beta = r.x
        data['status'].append(r.message)
        data['alpha'].append(alpha)
        data['beta'].append(beta)
        data['subject'].append(pkl[:2])
        fig, ax = plt.subplots(1, 1)
        plot_single_subject(pkl, ax, r)
        figs.append(fig)
        plt.close()
    cols = ('subject', 'alpha', 'beta', 'status')
    df = pd.DataFrame(data, columns=cols)
    df.to_csv('fit.csv')
    save_figs_as_pdf(figs, 'nllf.pdf')
    

def fit_single_subject(subject_number):
    fn = '{:0>2d}.pkl'.format(subject_number)
    if os.path.isfile(os.path.join(Data_Behavior_Dir, fn)) is False:
        print('No data for subject', subject_number)
        return 
    r = fit_model(fn)
    print(r)
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    plot_single_subject(fn, ax, r)
    plt.show()

if __name__ == '__main__':
    fit_single_subject(int(sys.argv[1]))
