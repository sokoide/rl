from stable_baselines3.common.atari_wrappers import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import time
import gym

ENV_ID = 'MsPacman-v0'
NUM_ENV = 8
STEPS = 5_000

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = EpisodicLifeEnv(env)
        env.seed(seed + rank)
        return env
    return _init

def main():
    env = DummyVecEnv([make_env(ENV_ID, i) for i in range(NUM_ENV)])
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=STEPS)
    model.save(ENV_ID)
    del model

    model = PPO.load(ENV_ID, env=env, verbose=1)
    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        time.sleep(1/60)

if __name__ == "__main__":
    main()
