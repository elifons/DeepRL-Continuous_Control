from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from utils.ddpg_agent import Agent
import argparse
import os
import os.path as op

# Code based on https://github.com/udacity/deep-reinforcement-learning.git


def ddpg(dir_, n_episodes=500, max_t=1000, learn_every=20, num_learning=10, goal=30.0):
    scores_window = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()                
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]     # send the action to the environment
            next_state = env_info.vector_observations   # get the next state
            reward = env_info.rewards                   # get the reward
            done = env_info.local_done                  # see if episode has finished            
            
            for s, a, r, n_s, d in zip(state, action, reward, next_state, done):
                agent.add_memory(s, a, r, n_s, d)

            state = next_state
            score += env_info.rewards #reward

            if t % learn_every == 0:
                for _ in range(num_learning):
                    agent.step()
            
            if np.any(done):
                break 
                
        mean_score = np.mean(score)      
        scores_window.append(mean_score)
        scores.append(mean_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), mean_score), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), op.join(dir_, 'checkpoint_actor.pth'))
            torch.save(agent.critic_local.state_dict(), op.join(dir_, 'checkpoint_critic.pth'))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))   
        if np.mean(scores_window)>=goal and i_episode >= 120:
            print('\nEnvironment solved after {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), op.join(dir_, 'checkpoint _actor.pth'))
            torch.save(agent.critic_local.state_dict(), op.join(dir_, 'checkpoint_critic.pth'))
            break
    return scores

def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass

if __name__ == '__main__':  
    # Inputs for the main function
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_episodes', default=500, type=int, help='max number of training episodes')
    parser.add_argument('--max_t', default=1000, type=int, help='max. number of timesteps per episode')
    parser.add_argument('--learn_every', default=20, type=int, help='number of timesteps to wait until updating network')
    parser.add_argument('--num_learning', default=10, type=int, help='number of updates')
    parser.add_argument('--goal', default=30.0, type=float, help='reward goal that considers the problem solved')  
    parser.add_argument('--seed', default=7, type=int, help='training seed')  
    parser.add_argument('--dest', default='runs', type=str, help='experiment dir')
    args = parser.parse_args() 

    # Change the file_name parameter to match the location of the Univy environment.
    env = UnityEnvironment(file_name='Reacher_single.app')
    path = args.dest
    create_directory(path)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]


    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


    agent = Agent(state_size=state_size, action_size=action_size, random_seed=args.seed)

    scores = ddpg(dir_=path, n_episodes=args.n_episodes, max_t=args.max_t, learn_every=args.learn_every, num_learning=args.num_learning, goal=args.goal)

    env.close()

    # plot scores
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(op.join(path, 'scores_values.csv'))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(op.join(path, 'score.png'))
