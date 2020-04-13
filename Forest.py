from hiive.mdptoolbox import example
from hiive.mdptoolbox import mdp
import numpy as np
import util
import time
import pprint

q_counter = 0

def render(nS, rewards, transitions, policy):

    # observation = np.random.choice(nS, 1, transitions[action, ])
    # for i, action in enumerate(policy):
    #     observation = np.random.choice(nS, 1, transitions[action, i,  ])
    # reward = [action[]]

    return
def callback(state,action,new_state):
    # print(state,action,new_state)
    global q_counter
    q_counter += 1
    return

def run(verbose=False):
    # MDP Forest Problem 
    # transitions, reward = example.forest()
    nS = 1000
    # transitions, reward = example.forest(S=nS, r1=250, r2=120, p=0.01, is_sparse=False)
    transitions, reward = example.forest(S=nS, r1=1045, r2=1025, p=0.01, is_sparse=False)

    # print(transitions)
    # print (reward)
    # return
    print('~~~~~~~~~~ Forest - Policy Iteration ~~~~~~~~~~')
    pi = mdp.PolicyIteration(transitions, reward, 0.75, max_iter=10000)
    if verbose:
        pi.setVerbose()
    pi.run()
    util.print_debugs(pi)
    # print(pi.run_stats)
    # return

    print('~~~~~~~~~~ Forest - Value Iteration ~~~~~~~~~~')
    vi = mdp.ValueIteration(transitions, reward, 0.75, max_iter=100000)
    if verbose:
        vi.setVerbose()
    vi.run()
    util.print_debugs(vi)

    if(vi.policy == pi.policy):
        print('Forest - Value and Policy Iteration policies are the same! ')
    else:
        print('Forest - Value and Policy Iteration policies are NOT the same.')

    print('~~~~~~~~~~ Forest - Q-Learning ~~~~~~~~~~')
    # transitions, reward, gamma,
                    #  alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
                    #  epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99,
                    #  n_iter=10000, skip_check=False, iter_callback=None,
                    #  run_stat_frequency=None):
                    
    ql = mdp.QLearning(transitions, reward, 0.75, alpha=0.3, epsilon_min=0.005, n_iter=500000)
    if verbose:
        ql.setVerbose()
    start_t = time.process_time()
    ql.run()
    end_t = time.process_time()

# Output
    print('~~~~~~~~~~ Forest - Policy Iteration ~~~~~~~~~~')
    util.print_debugs(pi)
    print('~~~~~~~~~~ Forest - Value Iteration ~~~~~~~~~~')
    util.print_debugs(vi)
    print('~~~~~~~~~~ Forest - Q-Learning ~~~~~~~~~~')
    print(ql.policy)
    print('Q-Learning # of Iterations: %i' %q_counter) 
    print('Clock time')
    print(end_t - start_t)

    if(vi.policy == pi.policy):
        print('Forest - Value and Policy Iteration policies are the same! ')
    else:
        print('Forest - Value and Policy Iteration policies are NOT the same.')
    
    
    if(vi.policy == ql.policy):
        print('Forest – QL and VI Policies are the same!')
    else:
        print('Forest – QL and VI Policies are NOT the same.')
    if(pi.policy == ql.policy):
        print('Forest – PI and PI Policies are the same!')
    else:
        print('Forest – PI and VI Policies are NOT the same.')

    # A Q-Learning Algorithm
    # 
    # Source: 
    #   https://www.oreilly.com/radar/introduction-to-reinforcement-learning-and-openai-gym/
    """
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    G = 0
    alpha = 0.618
    for episode in range(1,1001):
        done = False
        G, reward = 0,0
        state = env.reset()
        while done != True:
                action = np.argmax(Q[state]) #1
                state2, reward, done, info = env.step(action) #2
                Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
                G += reward
                state = state2   
        if episode % 50 == 0:
            print('Episode {} Total Reward: {}'.format(episode,G))
    print(Q)
    """

    # mdp.ValueIteration(transitions, reward, discount, epsilon=0.01, 
    # max_iter=1000, initial_value=0, skip_check=False)