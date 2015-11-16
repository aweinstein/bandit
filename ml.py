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
from collections import defaultdict
from matplotlib import cm

from utils import save_figs_as_pdf, softmax
from models import Bandit, BanditCard, BanditCardCues
from models import Agent, AgentCard, AgentCardCues

Data_Behavior_Dir = 'data_behavior'
Fig_Dir = 'figs'
DF_Dir = 'df'

class ML(object):
    def __init__(self, df, n_actions, cues=None):
        """The DataFrame df must contain columns 'action' and reward'.
        If `len(cues) > 1`, then it also must include the 'cue' column.
        """
        self.n_actions = n_actions
        if cues is None:
            if 'cue' in df.columns:
                self.cues = (df['cue'].values[0],)
                print('Using {:d} for the cue.'.format(self.cues[0]))
            else:
                self.cues = (0,)
                df['cue'] = 0
        else:
            self.cues = cues
        if type(cues) is int:
            self.cues = (cues,)
        self.df = df
        
    def neg_log_likelihood(self, alphabeta):
        df = self.df
        alpha, beta = alphabeta
        df = self.df[self.df['cue'].isin(self.cues)]
        actions, rewards = df['action'].values, df['reward'].values
        cues = df['cue'].values
        prob_log = 0
        Q = dict([[cue, np.zeros(self.n_actions)] for cue in self.cues])
        for action, reward, cue in zip(actions, rewards, cues):
            Q[cue][action] += alpha * (reward - Q[cue][action])
            prob_log += np.log(softmax(Q[cue], beta)[action])
        return -prob_log

    def ml_estimation(self, method_name='Nelder-Mead', bounds=None):
        if bounds is None:
            r = minimize(self.neg_log_likelihood, [0.1,0.1],
                         method=method_name)
        else:
            r = minimize(self.neg_log_likelihood, [0.1,0.1],
                              method='L-BFGS-B',
                              bounds=bounds)
        return r

    def fit_model(self, bounds=None):
        r = self.ml_estimation('Nelder-Mead', bounds)
        if r.status != 0:
            print('trying with Powell')
            r = self.ml_estimation('Powell', bounds)
        return r

    def plot_ml(self, ax, alpha, beta, alpha_hat, beta_hat):
        from itertools import product
        n = 50
        alpha_max = 0.2
        beta_max = 1.3
        if alpha is not None:
            alpha_max = alpha_max if alpha < alpha_max else 1.1 * alpha
            beta_max = beta_max if beta < beta_max else 1.1 * beta
        if alpha_hat is not None:
            alpha_max = alpha_max if alpha_hat < alpha_max else 1.1 * alpha_hat
            beta_max = beta_max if beta_hat < beta_max else 1.1 * beta_hat
        alphas = np.linspace(0, alpha_max, n)
        betas = np.linspace(0, beta_max, n)
        Alpha, Beta = np.meshgrid(alphas, betas)
        Z = np.zeros(len(Alpha) * len(Beta))
        for i, (a, b) in enumerate(product(alphas, betas)):
            Z[i] = self.neg_log_likelihood((a, b))
        Z.resize((len(alphas), len(betas)))
        ax.contourf(Alpha, Beta, Z.T, 50, cmap=cm.jet)
        if alpha is not None:
            ax.plot(alpha, beta, 'rs', ms=5)
        if alpha_hat is not None:
            ax.plot(alpha_hat, beta_hat, 'r+', ms=10)
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'$\beta$', fontsize=20)
        return
    
    def plot_single_subject(self, subject, ax, r):
        if r.status == 0:
            alpha, beta = r.x
            self.plot_ml(ax, alpha, beta, None, None)
            title = 'Subject {:d}'.format(subject)
        else:
            self.plot_ml(ax, None, None, None, None)
            title = 'Subject {:d}, not converged'.format(subject)
        ax.set_title(title)


def simple_bandit_experiment():
    b = Bandit()
    alpha = 0.1
    beta = 1.2
    print('alpha: {:.2f} beta: {:.2f}\n'.format(alpha, beta))
    agent = Agent(b, alpha, beta)
    trials = 1000

    for _ in range(trials):
        agent.run()
    df = agent.get_df()
    ml = ML(df, 2)
    r = ml.ml_estimation()
    alpha_hat, beta_hat = r.x
    print(r)
    fig, ax = plt.subplots(1, 1)
    ml.plot_ml(ax, alpha, beta, alpha_hat, beta_hat)
    plt.show()

def card_bandit_experiment():
    b = BanditCard()
    alpha = 0.2
    beta = 0.5
    print('alpha: {:.2f} beta: {:.2f}\n'.format(alpha, beta))
    agent = AgentCard(b, alpha, beta)
    trials = 360

    for _ in range(trials):
        agent.run()
    df = agent.get_df()
    df.to_csv('data.csv', index_label='trial')
    print('Total reward: {:d}\n'.format(df['reward'].sum()))
    ml = ML(df, 4)
    r = ml.ml_estimation()
    alpha_hat, beta_hat = r.x
    print(r)

    fig, ax = plt.subplots(1, 1)
    ml.plot_ml(ax, alpha, beta, alpha_hat, beta_hat)
    plt.show()
    globals().update(locals())

def card_cue_bandit_experiment():
    b = BanditCardCues()
    alpha = 0.2
    beta = 0.5
    print('alpha: {:.2f} beta: {:.2f}\n'.format(alpha, beta))
    agent = AgentCardCues(b, alpha, beta)
    trials = 360*3

    for _ in range(trials):
        agent.run()
    df = agent.get_df()
    df.to_csv('data.csv', index_label='trial')

    ml = ML(df, 4, (0,1))
    r = ml.ml_estimation()
    print(r)

    alpha_hat, beta_hat = r.x
    fig, ax = plt.subplots(1, 1)
    ml.plot_ml(ax, alpha, beta, alpha_hat, beta_hat)
    plt.show()
    globals().update(locals())


def fit_behavioral_data(bounds=None, cue=0):
    """Fit a model for all subjects.

    The data has been previously parsed by parse.py.
    """
    pkls = os.listdir(Data_Behavior_Dir)
    pkls.sort()
    data = {'alpha':[], 'beta':[], 'subject': [], 'status':[]}
    figs = []
    for pkl in pkls:
        print(pkl)
        df = pd.read_pickle(os.path.join(Data_Behavior_Dir, pkl))
        df = df[df['cue']==cue]
        ml = ML(df, 4)
        r = ml.fit_model(bounds)
        alpha, beta = r.x
        data['status'].append(r.message)
        data['alpha'].append(alpha)
        data['beta'].append(beta)
        data['subject'].append(pkl[:2])
        fig, ax = plt.subplots(1, 1)
        ml.plot_single_subject(int(pkl[:2]), ax, r)
        figs.append(fig)
        plt.close()
    cols = ('subject', 'alpha', 'beta', 'status')
    df = pd.DataFrame(data, columns=cols)
    df.to_csv('fit.csv')
    if bounds is None:
        fn = os.path.join(Fig_Dir, 'nllf_unbounded.pdf')
    else:
        fn = os.path.join(Fig_Dir, 'nllf_bounded.pdf')
    save_figs_as_pdf(figs, fn)

def fit_single_subject(subject_number, bounds=None, cue=0):
    fn = os.path.join(Data_Behavior_Dir, '{:0>2d}.pkl'.format(subject_number))
    if os.path.isfile(fn) is False:
        print('No data for subject', subject_number)
        return
    df = pd.read_pickle(fn)
    df = df[df['cue']==cue]
    ml = ML(df, 4)
    r = ml.fit_model(bounds)
    plt.close('all')
    fig, ax = plt.subplots(1, 1)
    ml.plot_single_subject(subject_number, ax, r)
    plt.show()
    return r

def get_learner_class(actions, opt_action):
    """Determine if the actions correspond to a learner behavior.

    Parameters
    ----------
    actions: ndArray or Series
        List of actions selected by the subject
    opt_action: int
        Optimal action

    Returns
    -------
    learner: bool
        True if the subject is a learner
    n_optimals: list
        List with the number of optimal actions in each segment of 20 trials

    Notes
    -----
    See the following for details

    T. Schonberg, N. D. Daw, D. Joel, and J. P. O'Doherty, "Reinforcement
    Learning Signals in the Human Striatum Distinguish Learners from
    Nonlearners during Reward-Based Decision Making," J. Neurosci., vol. 27,
    no. 47, pp. 12860–12867, Nov. 2007.
    """
    block_size = 20
    n_blocks = int(len(actions) / block_size)
    blocks = np.array_split(actions, n_blocks)
    n_optimals = [(block==opt_action).sum() for block in blocks]
    last_n_size = 40
    threshold = 25
    learner = (actions[-last_n_size:]==opt_action).sum() > threshold
    return learner, n_optimals

def make_learner_df():
    df = pd.read_pickle(os.path.join(DF_Dir, 'all_data.pkl'))
    cue = 1
    opt_choice = 23

    subjects = df.index.get_level_values('subject').unique()
    learners = {}
    n_optimum = defaultdict(list)

    for subject in subjects:
        actions = df[df['cue'] == cue].ix[subject]['action']
        learner , n_per_block = get_learner_class(actions, opt_choice)
        learners[subject] = learner
        for i, n in enumerate(n_per_block):
            n_optimum['subject'].append(subject)
            n_optimum['block'].append(i + 1)
            n_optimum['n_optimum'].append(n)
            n_optimum['learner'].append(learner)
    
    df_learners = pd.DataFrame(pd.Series(learners), columns=['learner'])
    df_learners.index.set_names('subject', inplace=True)
    cols = ['subject', 'block', 'n_optimum', 'learner']
    df_n_optimum = pd.DataFrame(n_optimum, columns=cols)
    return df_learners, df_n_optimum
    
if __name__ == '__main__':
    bounds = ((0,1), (0,2))
    fit_single_subject(int(sys.argv[1]), bounds)
