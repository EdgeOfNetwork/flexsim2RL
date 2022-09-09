import gym
from flexsim_env import FlexSimEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import file_config
import time
from datetime import timedelta


def main():
    start = time.process_time()
    print("Initializing FlexSim environment...")

    # Create a FlexSim OpenAI Gym Environment
    env = FlexSimEnv(
        flexsimPath= file_config.configs['flexsimPath'],
        modelPath = file_config.configs['modelPath'],
        verbose = False,
        visible = False
    )

    check_env(env) # Check that an environment follows Gym API.

    # Training a baselines3 PPO model in the environment
    model = PPO("MlpPolicy", env, verbose=1)
    print("Training model...")
    model.learn(total_timesteps=10000)
    
    # save the model
    print("Saving model...")
    model.save("flex_rl_model")

    input("Waiting for input to do some test runs...")

    # Run test episodes using the trained model
    for i in range(2):
        env.seed(i)
        observation = env.reset()
        env.render()
        done = False
        rewards = []
        while not done:
            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(action)
            env.render()
            rewards.append(reward)
            if done:
                cumulative_reward = sum(rewards)
                print("Reward: ", cumulative_reward, "\n")
    env._release_flexsim()
    end = time.process_time()
    print("Traing Time Elapsed: ", end - start)
    print("Traing Time Elapsed: ", timedelta(seconds = end - start))

    input("Waiting for input to close FlexSim...")
    env.close()


if __name__ == "__main__":
    main()