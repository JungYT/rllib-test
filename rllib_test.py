import numpy as np
import numpy.random as random
import ray
from ray import tune
from ray.rllib.agents import ddpg
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
import gym
from pathlib import Path

import fym
from fym.core import BaseEnv, BaseSystem
from postProcessing import plot_rllib_test, plot_compare


class Plant(BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = BaseSystem(shape=(2,1))
        self.vel = BaseSystem(shape=(2,1))

    def set_dot(self, u):
        self.pos.dot = self.vel.state
        self.vel.dot = u


class MyEnv(BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config)
        self.plant = Plant()

        self.action_space = gym.spaces.Box(low=-10., high=10., shape=(2,))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.plant.state.shape
        )

    def reset(self, initial="random"):
        if initial == "random":
            self.plant.initial_state = 5 * (
                2*random.rand(*self.plant.state.shape) - 1
            )
        else:
            self.plant.initial_state = initial
        super().reset()
        obs = self.observe()
        return obs

    def step(self, action):
        u = np.vstack(action)
        *_, done = self.update(u=u)
        obs = self.observe()
        reward = self.get_reward(u)
        info = {}
        return obs, reward, done, info

    def set_dot(self, t, u):
        self.plant.set_dot(u)
        x = np.float32(self.plant.state)
        pos = x[0:2]
        vel = x[2:4]
        lyap_dot = pos.squeeze() @ vel.squeeze()
        return dict(t=t, **self.observe_dict(), action=u, lyap_dot=lyap_dot)

    def observe(self):
        obs = np.float32(self.plant.state)
        return obs

    def get_reward(self, u):
        x = np.float32(self.plant.state)
        pos = x[0:2]
        vel = x[2:4]
        lyap = pos.squeeze() @ pos.squeeze()
        lyap_dot = pos.squeeze() @ vel.squeeze()
        # reward = -lyap_dot
        # if lyap_dot <= -0.1*lyap:
        #     reward = -1
        # else:
        #     reward = -10
        # if lyap_dot <= 0:
        #     reward = -3
        # else:
        #     reward = -6

        # if lyap < 1e-5:
        #     reward = 1
        # tmp = np.float32(0.5*np.exp(-(pos.T @ pos).item()))
        reward = np.float32(
            np.exp(
                1e-1 * (
                    -x.T @ np.diag([100, 100, 1, 1]) @ x 
                    - u.T @ np.diag([10, 10]) @ u
                ).item()
            )
        )
        # reward += tmp
        # reward = np.float32(
        #     (-x.T@np.diag([1, 1, 0, 0])@x 
        #      - u.T@np.diag([0, 0])@u).item()
        # )
        # reward = -5e-3 * np.linalg.norm(pos).item() \
        #     - 1e-5 * np.linalg.norm(vel).item()

        return reward


def train():
    cfg = fym.config.load(as_dict=True)

    analysis = tune.run(ppo.PPOTrainer, **cfg)
    parent_path = Path(analysis.get_last_checkpoint(
        metric="episode_reward_mean",
        mode="max"
    )).parent.parent
    checkpoint_paths = analysis.get_trial_checkpoints_paths(
        trial=str(parent_path)
    )
    
    return checkpoint_paths


# @ray.remote
@ray.remote(num_cpus=6)
def sim(initial, checkpoint_path, env_config, num=0):
    # env_config = fym.config.load("config.env_config", as_dict=True)
    agent = ppo.PPOTrainer(env=MyEnv, config={"explore": False})
    agent.restore(checkpoint_path)

    env = MyEnv(env_config)
    parent_path = Path(checkpoint_path).parent
    data_path = Path(parent_path, f"test_{num+1}", "env_data.h5")
    plot_path = Path(parent_path, f"test_{num+1}")
    
    env.logger = fym.Logger(data_path)

    obs = env.reset(initial)
    total_reward = 0
    while True:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()
    # print(initial)
    # print(f"finish {num+1} episode")
    plot_rllib_test(plot_path, data_path)


def multi_validate(initials, checkpoint_paths, env_config):
    futures = [sim.remote(initial, path[0], env_config, num=i)
               for i, initial in enumerate(initials)
               for path in checkpoint_paths]
    ray.get(futures)


def validate(parent_path):
    _, info = fym.logging.load(Path(parent_path, 'checkpoint_paths.h5'), with_info=True)
    checkpoint_paths = info['checkpoint_paths']
    initials = []
    print("Making initials...")
    random.seed(0)
    while True:
        tmp = 20 * (2 * np.random.rand(4,1) -1)
        if np.all([
                np.sqrt(np.sum(tmp[:2, :]**2, axis=0)) < 10,
                np.sqrt(np.sum(tmp[:2, :]**2, axis=0)) > 5,
                np.sqrt(np.sum(tmp[2:, :]**2, axis=0)) < 3,
        ], axis=0):
            initials.append(tmp)
        if len(initials) == 5:
            break
    fym.config.update({"config.env_config.max_t": 20})
    env_config = ray.put(fym.config.load("config.env_config", as_dict=True))
    print("Validating...")
    multi_validate(initials, checkpoint_paths, env_config)

def plot_data(parent_path_list):
    for parent_path in parent_path_list:
        _, info = fym.logging.load(Path(parent_path, 'checkpoint_paths.h5'), with_info=True)
        checkpoint_paths = info['checkpoint_paths']
        # for checkpoint_data in checkpoint_paths:
        checkpoint_path = Path(checkpoint_paths[-1][0]).parent
        test_path_list = [x for x in checkpoint_path.iterdir() if x.is_dir()]
        # for i in range(len(test_path_list)):
        data_path = Path(test_path_list[-1], "env_data.h5")
        plot_path = Path(test_path_list[-1])
        print("Ploting", str(plot_path))
        plot_rllib_test(plot_path, data_path)

def main():
    fym.config.reset()
    fym.config.update({
        "config": {
            "env": MyEnv,
            "env_config": {
                "dt": 0.01,
                "max_t": 10.,
                "solver": "rk4"
            },
            "num_gpus": 0,
            "num_workers": 4,
            # "num_envs_per_worker": 50,
            "lr": 0.0001,
            "gamma": 0.99,
            # "lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "gamma": tune.grid_search([0.9, 0.99, 0.999])
            # "actor_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "critic_lr": tune.grid_search([0.001, 0.003, 0.0001]),
            # "actor_lr": 0.001,
            # "critic_lr": 0.0001,
            # "gamma": tune.grid_search([0.9, 0.99, 0.999, 0.9999]),
            # "exploration_config": {
            #     "random_timesteps": 10000,
            #     "scale_timesteps": 100000,
            # },
        },
        "stop": {
            "training_iteration": 1000,
        },
        "local_dir": "./ray_results",
        "checkpoint_freq": 100,
        "checkpoint_at_end": True,
    })
    checkpoint_paths = train()
    parent_path = "/".join(checkpoint_paths[0][0].split('/')[0:8])
    checkpoint_logger = fym.logging.Logger(
        Path(parent_path, 'checkpoint_paths.h5')
    )
    checkpoint_logger.set_info(checkpoint_paths=checkpoint_paths)
    checkpoint_logger.set_info(config=fym.config.load(as_dict=True))
    checkpoint_logger.close()
    return parent_path


if __name__ == "__main__":
    ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    ## To train, validate, and make figure
    parent_path = main()
    ## To validate and make figure
    # parent_path = './ray_results//'
    validate(parent_path)
    ray.shutdown()

    # To only Make Figure
    # pathes = [
    #     './ray_results/PPO_2021-10-30_11-14-17/',
    #     './ray_results/PPO_2021-10-30_04-39-15/',
    #     './ray_results/PPO_2021-10-30_17-48-39/',
    #     './ray_results/PPO_2021-10-29_00-16-02/',
    #     './ray_results/PPO_2021-10-30_19-14-30/',
    #     './ray_results/PPO_2021-10-30_20-59-25/',
    #     './ray_results/PPO_2021-10-31_03-13-36/',
    #     './ray_results/PPO_2021-10-30_07-49-32/',
    #     './ray_results/PPO_2021-10-31_08-35-38/',
    # ]
    # plot_data(pathes)




