import yaml
import argparse

from Env.carla_env import CarlaEnv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Carla Env')
    parser.add_argument('--cfg_path', type=str, required=True, help='please input the config file path')
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    env = CarlaEnv(cfg)
    obs, info =env.reset()
    for i in range(200):
        # A Example with random action policy
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        print(info)
        if terminated or truncated:
            break

    env.close()