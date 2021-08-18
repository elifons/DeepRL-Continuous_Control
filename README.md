[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Reinforcement Learning: Actor-critic for Continuous Control

This project presents an actor-critic model (deep deterministic policy gradient - DDPG) that operates on continuous action spaces, in our case, within the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. In this environment, a double-jointed arm can move to target locations. 

![Trained Agent][image1]

### Installation

1. Clone repository:
```
$ git clone https://github.com/elifons/DeepRL-Continuous_Control.git 
$ cd DeepRL-Continuous_Control
$ pip install -r requirements.txt
```

Alternatively, follow the instractions on this link https://github.com/udacity/deep-reinforcement-learning#dependencies to set up a python environment.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
3. Place the file in the DeepRL-Continuous_Control directory, and unzip (or decompress) the file. 

### Environment

 A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. 

There are two versions of the environment, in the first version, one agent is used and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes. In the second version, the environment consists on 20 agents training in parallel, and to take into account the presence of many agents, the criteria to solve it is the following, the  agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting started

Example command to run the code.

```
$ python3 main.py --dest exp_ddpg --n_episodes 1000
```

Or you can follow the instructions in `Continuous_Control.ipynb` to get started with training the agent.

**optional arguments:**

```
  --n_episodes N_EPISODES		max number of training episodes (default: 500)
  --max_t MAX_T         		max. number of timesteps per episode (default: 1000)
  --learn_every LEARN_EVERY		number of timesteps to wait until updating network (default: 20)
  --num_learning NUM_LEARNING	number of updates (default: 10)
  --goal GOAL           		reward goal that considers the problem solved (default: 30.0)
  --seed SEED           		training seed (default: 7)
  --dest DEST           		experiment dir (default: runs)
```



