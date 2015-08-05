import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
    def __init__(self, bandit, epsilon=0.1):
        self.epsilon = epsilon
        self.bandit = bandit
        n =  bandit.n
        #self.Q = np.zeros(n)
        self.Q = np.random.uniform(0, 1e-4, n) # init with small number to
                                               # avoid ties
        self.rewards_seq = []
        self.actions = []
        self.k_actions = np.ones(n) # number of steps for each action
        self.k_reward = 1
        self.average_reward = 0

        self.k_total = 1
        self.k_optimal = 0
        
    #@profile
    def run(self):
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.bandit.n)
        else:
            action = np.argmax(self.Q)
        reward = self.bandit.reward(action)
        
        k = self.k_actions[action]
        self.Q[action] +=  (reward - self.Q[action]) / k
        self.k_actions[action] += 1

        self.rewards_seq.append(reward)
        self.actions.append(action)
        self.average_reward += (reward - self.average_reward) / self.k_reward
        self.k_reward += 1

        if action == self.bandit.optimal_action():
            self.k_optimal += 1
        self.k_total += 1
        self.percentage_optimal = (self.k_optimal / self.k_total) * 100

    def reset(self):
        self.rewards = []
        

def run_experiment(n_bandits, steps, epsilon):
    average_rewards = np.zeros((n_bandits, steps))
    percentage_optimals = np.zeros((n_bandits, steps))
    for i in range(n_bandits):
        bandit = Bandit()
        agent = Agent(bandit, epsilon)
        for j in range(steps):
            agent.run()
            average_rewards[i,j] = agent.average_reward
            percentage_optimals[i,j] = agent.percentage_optimal

    return average_rewards, percentage_optimals
         

if __name__ == '__main__':
    np.random.seed(1234)
    epsilons = (0.1, 0.01, 0)
    ars, pos = [], []
    for epsilon in epsilons:
        ar, po = run_experiment(1000, 1000, epsilon)
        ars.append(np.mean(ar, 0))
        pos.append(np.mean(po, 0))
        
    plt.close('all')
    f, (ax1, ax2) = plt.subplots(2)
    for i,epsilon in enumerate(epsilons):
        ax1.plot(ars[i].T, label='$\epsilon$=%.2f' % epsilon)
        ax2.plot(pos[i].T, label='$\epsilon$=%.2f' % epsilon)
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Plays')
    ax1.set_ylabel('Average reward')
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Plays')
    ax2.set_ylabel('% Optimal action')
    plt.show()
    
if __name__ == '__main__x':
    bandit = Bandit()
    agent = Agent(bandit, 0.1)
    trials = 1000
    for _ in range(trials):
        agent.run()

    print('bandit.q_star')
    print(bandit.q_star)
    print('agent.Q:')
    print(agent.Q)
    print([len(x) for x in agent.rewards])
    

    
