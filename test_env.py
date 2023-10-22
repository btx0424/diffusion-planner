import torch
# import d4rl
import gymnasium_robotics
from torchrl.envs import GymEnv
from torchrl.envs import ParallelEnv
from torchrl.collectors import SyncDataCollector


def main():
    env_name = "PointMaze_LargeDense-v3"
    num_envs = 4
    env = ParallelEnv(num_envs, lambda: GymEnv(env_name))
    collector = SyncDataCollector(
        env, 
        policy=None,
        frames_per_batch=num_envs * 100,
        total_frames=-1,
        return_same_td=True
    )

    for i, data in enumerate(collector):
        print(data)
        break


if __name__ == '__main__':
    maze_envs = [env for env in GymEnv.available_envs if "PointMaze_Large" in env]
    print(maze_envs)
    
    main()

