import gym
from pprint import pprint 
import numpy as np
from hiive.mdptoolbox import mdp
import util
import time

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Prints out each step of a policy for OpenAI Gym Env
def render_env_policy(env, policy, display=False):
    t=0
    total_reward = 0
    observation = env.reset() # initial state
    for action in policy:
        t += 1
        if display:
            env.render()
        print(t)
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done and reward == 0.0:
            reward = -0.75
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.render()
    print('Total Reward: %f' %total_reward)
    return total_reward


# Generate transitions probability matrix for MDPToolbox
def transform_for_MDPToolbox(env):
    nA, nS = env.nA, env.nS
    P = np.zeros([nA, nS, nS])
    R = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for (p_trans, next_s, reward, done) in transitions:
                P[a,s,next_s] += p_trans
                if done and reward == 0.0:
                    reward = -0.75
                R[s,a] = reward
            P[a,s,:] /= np.sum(P[a,s,:])
    # pprint(P)
    return P, R

def print_policy(policy):
    for i, action in enumerate(policy):
        if action == LEFT:
            print(i, ' LEFT')
        elif action == RIGHT: 
            print(i, ' RIGHT')
        elif action == DOWN:
            print(i, ' DOWN')
        elif action == UP:
            print(i, ' UP')

def run(verbose=False):
    # env = gym.make('FrozenLake-v0', is_slippery=True)
    env = gym.make('FrozenLake8x8-v0', is_slippery=True)
    # env = gym.make('FrozenLake-v0')

    # Debug
    # print('env.P')
    # pprint(env.P)
    # print('env.R')
    # print(env.R)

    

    P, R = transform_for_MDPToolbox(env)
    # print('Reward')
    # print(R)
    # return

    print('~~~~~~~~~~ FrozenLake-v0 – 4x4 Policy Iteration ~~~~~~~~~~')
    pi = mdp.PolicyIteration(P, R, 0.6, max_iter=100000)

    if verbose:
        pi.setVerbose()
    pi.run()
    util.print_debugs(pi)
    total_r_pi = render_env_policy(env, pi.policy, display=verbose)


    print('~~~~~~~~~~ FrozenLake-v0 – 4x4 Value Iteration ~~~~~~~~~~')
    vi = mdp.ValueIteration(P, R, 0.6, epsilon=0.005, max_iter=10000)
    if verbose:
        vi.setVerbose()
    vi.run()
    util.print_debugs(vi)
    total_r_vi = render_env_policy(env, pi.policy, display=verbose)
    if(vi.policy == pi.policy):
        print('FrozenLake-v0 4x4 - Value and Policy Iteration policies are the same! ')
    else:
        print('FrozenLake-v0 4x4 - Value and Policy Iteration policies are NOT the same. ')


    print('~~~~~~~~~~ FrozenLake-v0 – Q-Learning ~~~~~~~~~~')
    ql = mdp.QLearning(P, R, 0.6, alpha=0.3, epsilon_min=0.005, n_iter=100000)
    if verbose:
        ql.setVerbose()
    start_t = time.process_time()
    ql.run()
    end_t = time.process_time()

    total_r_ql = render_env_policy(env, ql.policy, display=verbose)

# Output
    print('~~~~~~~~~~ FrozenLake-v0 - Policy Iteration ~~~~~~~~~~')
    util.print_debugs(pi)
    print('Total Reward: %f' %total_r_pi)
    print('~~~~~~~~~~ FrozenLake-v0 - Value Iteration ~~~~~~~~~~')
    util.print_debugs(vi)
    print('Total Reward: %f' %total_r_vi)
    print('~~~~~~~~~~ FrozenLake-v0 - Q-Learning ~~~~~~~~~~')
    print('Clock time')
    print(end_t - start_t)
    print('Total Reward: %f' %total_r_pi)
    print(ql.policy)
    
    if(vi.policy == pi.policy):
        print('FrozenLake-v0 - Value and Policy Iteration policies are the same! ')
    else:
        print('FrozenLake-v0 - Value and Policy Iteration policies are NOT the same.')
    
    
    if(vi.policy == ql.policy):
        print('FrozenLake-v0 – QL and VI Policies are the same!')
    else:
        print('FrozenLake-v0 – QL and VI Policies are NOT the same.')
    if(pi.policy == ql.policy):
        print('FrozenLake-v0 – PI and PI Policies are the same!')
    else:
        print('FrozenLake-v0 – PI and VI Policies are NOT the same.')

    print('VI Policy')
    print_policy(vi.policy)
    # print('PI Policy')
    # print_policy(vi.policy)
    print('QL Policy')
    print_policy(ql.policy)


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