import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from ppo import PPO, Normalization, RewardScaling
# from motion_planning_env import MotionPlanningEnv


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(para):

    """ 
    MODELS_INFO = {
        "X1-Y2-Z1": {
            "home": [0.264589, -0.293903, 0.777] 
        },
        "X2-Y2-Z2": {
            "home": [0.277866, -0.724482, 0.777] 
        },
        "X1-Y3-Z2": {
            "home": [0.268053, -0.513924, 0.777]  
        },
        "X1-Y2-Z2": {
            "home": [0.429198, -0.293903, 0.777] 
        },
        "X1-Y2-Z2-CHAMFER": {
            "home": [0.592619, -0.293903, 0.777]  
        },
        "X1-Y4-Z2": {
            "home": [0.108812, -0.716057, 0.777] 
        },
        "X1-Y1-Z2": {
            "home": [0.088808, -0.295820, 0.777] 
        },
        "X1-Y2-Z2-TWINFILLET": {
            "home": [0.103547, -0.501132, 0.777] 
        },
        "X1-Y3-Z2-FILLET": {
            "home": [0.433739, -0.507130, 0.777]  
        },
        "X1-Y4-Z1": {
            "home": [0.589908, -0.501033, 0.777]  
        },
        "X2-Y2-Z2-FILLET": {
            "home": [0.442505, -0.727271, 0.777] 
        }
    }
    env = MotionPlanningEnv(MODELS_INFO)
    """
    
    env = gym.make('CartPole-v1')
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n

    agent = PPO(para, para.STEP, state_dim, action_dim)

    state_norm = Normalization(dimension=state_dim)
    reward_norm = Normalization(dimension=1)
    reward_scaling = RewardScaling(dimension=1, gamma=para.gamma)
    
    update_step, total_step = 0, 0
    reward_episode = []

    for episode in range(para.EPISODE):
        reward_step = []
        state = env.reset(seed=para.SEED)
        state = state_norm(state[0]) if para.use_state_norm else state[0]
        reward_scaling.reset() if para.use_reward_scaling else None

        for step in range(para.STEP):
            action, action_prob = agent.choose_action(state)
            state_value = agent.get_state_value(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            reward_step.append(reward)

            next_state = state_norm(next_state) if para.use_state_norm else next_state
            reward = reward_norm(reward) if para.use_reward_norm else (reward_scaling(reward) if para.use_reward_scaling else reward)

            agent.replay_buffer.store(step, state, state_value, action, action_prob, reward, done, truncated)

            state = next_state
            total_step = total_step + 1
            if done: break

        if total_step > para.mini_batch_size and episode % para.update_frequency == 0:
            agent.update(update_step)
            update_step = update_step + 1

        #===================Save and Print Reward per Episode===================
        reward_episode.append(np.sum(reward_step))
        print('Episode:{}, Reward:{}'.format(episode, np.sum(reward_step)))
        #===================Save and Print Reward per Episode===================

    np.save('reward_episode.npy', reward_episode)


if __name__ == '__main__':
    env_para = {
        'SEED': 1,
        'EPISODE': 10000,
        'STEP': 200,
    }

    agent_para = {
        'update_frequency': 2,  # update network every 'update_frequency' episodes
        'max_train_steps': 2e6,  # Maximum number of training steps, used for lr decay.
        'batch_size': 64,  # Batch size
        'mini_batch_size': 16,  # Minibatch size
        'hidden_layer': [32, 32, 32],  # The number of neurons in hidden layers of the neural network
        'learning_rate': 0.001,  # Learning rate
        'gamma': 0.99,  # Discount factor
        'lamda': 0.99,  # GAE parameter
        'epsilon': 0.2,  # PPO clip parameter
        'K_epochs': 1,  # PPO parameter
        'use_adv_norm': True,  # Trick 1: advantage normalization
        'use_state_norm': True,  # Trick 2: state normalization
        'use_reward_norm': False,  # Trick 3: reward normalization
        'use_reward_scaling': True,  # Trick 4: reward scaling
        'entropy_coef': 0.01,  # Trick 5: policy entropy
        'use_lr_decay': True,  # Trick 6: learning rate Decay
        'use_grad_clip': True,  # Trick 7: gradient clip
        'use_orthogonal_init': True,  # Trick 8: orthogonal initialization
        'set_adam_eps': True,  # Trick 9: set Adam epsilon=1e-5
        'use_tanh': True,  # Trick 10: tanh activation function
        'use_value_clip': True,  # Whether to use value clip
    }
    para = SimpleNamespace(**env_para, **agent_para)
    seed_everywhere(para.SEED)
    train(para)
    plt.plot(np.load('reward_episode.npy'))
    plt.show()
    