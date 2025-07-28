import yaml
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from Env.carla_env import CarlaEnv


def train(cfg):
    # Initialize Environment
    env = CarlaEnv(cfg)

    # Initialize Agent
    if cfg['robot']['agent'] == "PPO_Agent":
        action_dims = cfg['train']['ppo_param']['action_dims']
        actor_lr = cfg['train']['ppo_param']['actor_lr']
        value_lr = cfg['train']['ppo_param']['value_lr']
        total_episodes_num = int(cfg['train']['total_episodes_num'])
        batch_size = cfg['train']['ppo_param']['batch_size']
        eps = cfg['train']['ppo_param']['eps']

        from Agent.PPO.ppo import PPO_Agent
        agent = PPO_Agent(
            action_space_dims=action_dims,
            total_episodes_num = total_episodes_num,
            batch_size=batch_size,
            actor_lr=actor_lr,
            value_lr=value_lr,
            eps=eps
        )
    
    # Training
    writer = SummaryWriter('runs/ppo')

    for episode in range(total_episodes_num):
        # Record List
        done = False
        states = []
        actions = []
        probs = []
        rewards = []

        state, info = env.reset()
        while not done:
            action, action_prob = agent.get_action(state, training=True)
            print('action: ', action, '  action prob: ', action_prob)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            probs.append(action_prob)
            rewards.append(reward)

            done = terminated or truncated
            state = next_state

            print(info)
        
        # Calculate cumulative rewards
        g = 0
        gamma = 0.99
        returns = []

        for reward in reversed(rewards):
            g = reward + gamma * g
            returns.insert(0, g)

        # Updata
        agent.train(states, actions, probs, returns)
        
        # Logging and saving weights
        writer.add_scalar('semantic/5-10m', sum(rewards), episode)
        
        if (episode + 1) % 100 == 0:
            torch.save(agent.actor_net.state_dict(), f"weights/ppo/semantic/actor_5_10m_{episode+1}_episode.pth")
            torch.save(agent.value_net.state_dict(), f"weights/ppo/semantic/value_5_10m_{episode+1}_episode.pth")

    env.close()
    



def test(cfg):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Test in Carla Environment')
    parser.add_argument('--cfg_path', type=str, required=True, help='please input the config file path')
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg['env']['mode'] == 'train':
        train(cfg)
    elif cfg['env']['mode'] == 'test':
        test(cfg)
    else:
        raise ValueError("Invalid mode in configuration. Use 'train' or 'test'.")
    
