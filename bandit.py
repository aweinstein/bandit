import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 2

class Bandit(object):
    def __init__(self, n=10):
        self.n = n
        self.q_star = np.random.normal(size=n)

    def reward(self, action):
        if action >= 0 and action < self.n:
            return self.q_star[action] + np.random.normal()
        else:
            print('Error: action out of range')

    def optimal_action(self):
        return np.argmax(self.q_star)

class Agent(object):
    def __init__(self, bandit, epsilon=0.1, Q_init=None, alpha=None):
        self.epsilon = epsilon
        self.bandit = bandit
        n =  bandit.n
        self.n = n
        if Q_init:
            self.Q = Q_init * np.ones(n)
        else:
            self.Q = np.random.uniform(0, 1e-4, n) # init with small numbers to
                                                   # avoid ties
        if alpha:
            self.update_action_value = self.update_action_value_constant_alpha
            self.alpha = alpha
        else:
            self.update_action_value = self.update_action_value_sample_average
        self.reset()
        
    def run(self):
        # Choose action
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.bandit.n)
        else:
            action = np.argmax(self.Q)
        reward = self.bandit.reward(action)
        
        # Update action-value
        self.update_action_value(action, reward)
 
        # Keep track of performance
        self.rewards_seq.append(reward)
        self.actions.append(action)
        self.average_reward += (reward - self.average_reward) / self.k_reward
        self.k_reward += 1
        self.optimal_actions.append(action == self.bandit.optimal_action())

    def update_action_value_sample_average(self, action, reward):
        k = self.k_actions[action]
        self.Q[action] +=  (1 / k) * (reward - self.Q[action])
        self.k_actions[action] += 1

    def update_action_value_constant_alpha(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])

    def reset(self):
        self.rewards = []
        self.rewards_seq = []
        self.actions = []
        self.k_actions = np.ones(self.n) # number of steps for each action
        self.k_reward = 1
        self.average_reward = 0
        self.optimal_actions = []


def run_experiment(n_bandits, steps, epsilon, Q_init=None, alpha=None):
    '''Run a 10-bandit simulation many times.

    Parameters
    ----------
    n_bandits : int
        Number of 10-armed bandits to run.
    steps : int
        Number of steps to run each simulation.
    epsilon : float
        Epsilon used by the agent.
    Q_init : float, optional
        Initial action-value estimate.
    alpha : float, optional
        Value of alpha to use if constant update rule is used
    Returns
    -------
    average_rewards: array_like
        Cumulative reward as function of the step, averaged over all the
        `n_bandits` simulations.
    percentage_optimals: array_like
        Cumulative percentage of times that the optimal actio is selected,
        averaged over all the `n_bandits` simulations.
    '''
    average_rewards = np.zeros((n_bandits, steps))
    percentage_optimals = np.zeros((n_bandits, steps))
    for i in range(n_bandits):
        bandit = Bandit()
        agent = Agent(bandit, epsilon, Q_init)
        for j in range(steps):
            agent.run()
        average_rewards[i,:] = agent.rewards_seq
        percentage_optimals[i,:] = agent.optimal_actions
    return average_rewards, percentage_optimals
         

def figure_2_1():
    '''Replicate figure 2.1 of Sutton and Barto's book.'''
    np.random.seed(1234)
    epsilons = (0.1, 0.01, 0)
    ars, pos = [], []
    for epsilon in epsilons:
        ar, po = run_experiment(2000, 1000, epsilon)
        ars.append(np.mean(ar, 0))
        pos.append(np.mean(po, 0))
        
    # plot the results
    plt.close('all')
    f, (ax1, ax2) = plt.subplots(2)
    for i,epsilon in enumerate(epsilons):
        ax1.plot(ars[i].T, label='$\epsilon$=%.2f' % epsilon)
        ax2.plot(pos[i].T, label='$\epsilon$=%.2f' % epsilon)
    ax1.legend(loc='lower right')
    ax1.set_ylabel('Average reward')
    ax1.set_xlim(xmin=-10)
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Plays')
    ax2.set_ylabel('% Optimal action')
    ax2.set_xlim(xmin=-20)
    plt.savefig('fig_2_1.pdf')
    plt.show()

def figure_2_4():
    '''Replicate figure 2.4 of Sutton and Barto's book.'''
    np.random.seed(1234)
    epsilons = (0.1, 0)
    q_inits = (False, 5)
    ars, pos = [], []
    for epsilon, q_init in zip(epsilons, q_inits):
        ar, po = run_experiment(100, 8000, epsilon, q_init, alpha=0.1)
        ars.append(np.mean(ar, 0))
        pos.append(np.mean(po, 0))
        
    # plot the results
    plt.close('all')
    plt.plot(pos[0], label='realistic $\epsilon$-greedy')
    plt.plot(pos[1], label='optimistic greedy')
    plt.legend(loc='lower right')
    plt.xlabel('Plays')
    plt.ylabel('% Optimal action')
    plt.xlim(xmin=-20)
    plt.savefig('fig_2_4.pdf')
    plt.show()

if __name__ == '__main__':
    figure_2_1()
