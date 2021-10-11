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
from postProcessing import plot_rllib_test


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
        return dict(t=t, **self.observe_dict(), action=u)

    def observe(self):
        obs = np.float32(self.plant.state)
        return obs

    def get_reward(self, u):
        x = np.float32(self.plant.state)
        reward = np.float32(
            -np.exp(
                1e-6 * (
                    -x.T @ np.diag([100, 100, 1, 1]) @ x 
                    - u.T @ np.diag([10, 10]) @ u
                ).item()
            )
        )
        return reward


def train():
    cfg = fym.config.load(as_dict=True)

    analysis = tune.run(ddpg.DDPGTrainer, **cfg)
    parent_path = Path(analysis.get_last_checkpoint(
        metric="episode_reward_mean",
        mode="max"
    )).parent.parent
    checkpoint_paths = analysis.get_trial_checkpoints_paths(
        trial=str(parent_path)
    )
    
    return checkpoint_paths


@ray.remote(num_cpus=8)
def validate(initial, checkpoint_path, num=1):
    env_config = fym.config.load("config.env_config", as_dict=True)
    agent = ddpg.DDPGTrainer(env=MyEnv, config={"explore": False})
    agent.restore(checkpoint_path)
    breakpoint()

    env = MyEnv(env_config)
    ext_path = Path(checkpoint_path, str(num), "env_data.h5")
    env.logger = fym.Logger(ext_path)

    obs = env.reset(initial[:, None])
    total_reward = 0
    while True:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    env.close()
    plot_rllib_test(ext_path, ext_path)


def multi_validate(initials, checkpoint_paths):
    futures = [validate.remote(initial, path[0], num=i)
               for i, initial in enumerate(initials)
               for path in checkpoint_paths]
    ray.get(futures)


def main():
    fym.config.reset()
    fym.config.update({
        "config": {
            "env": MyEnv,
            "env_config": {
                "dt": 0.05,
                "max_t": 20.,
                "solver": "odeint"
            },
            "num_gpus": 0,
            "num_workers": 6,
            # "lr": 0.001,
            # "gamma": 0.999
            "lr": 0.0001, ## DDPG
            "gamma": 0.9
            # "lr": tune.grid_search([0.001, 0.0001]),
            # "gamma": tune.grid_search([1.9, 0.99, 0.999]),
        },
        "stop": {
            "training_iteration": 2,
        },
        "local_dir": "./ray_results",
        "checkpoint_freq": 1,
        "checkpoint_at_end": True,
    })
    checkpoint_paths = train()
    initials = 3 * (2 * np.random.rand(100, 4) - 1)
    initials = initials[
        np.all([
            np.sqrt(np.sum(initials[:, :2]**2, axis=1)) < 3,
            np.sqrt(np.sum(initials[:, 2:]**2, axis=1)) < 3,
        ], axis=0)
    ]
    multi_validate(initials, checkpoint_paths)
    breakpoint()


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    main()
    ray.shutdown()




