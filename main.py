import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Use non-interactive backend
matplotlib.use('Agg')

device = 'cpu'  # CPU is usually faster for small DQN models


class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            hyperparameters = yaml.safe_load(file)[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(
            self.env_id,
            render_mode='human' if render else None,
            **self.env_make_params
        )

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_episode = []

        policy_dqn = DQN(
            num_states,
            num_actions,
            self.fc1_nodes,
            self.enable_dueling_dqn
        ).to(device)

        if is_training:
            epsilon = self.epsilon_init
            epsilon_history = []

            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(
                num_states,
                num_actions,
                self.fc1_nodes,
                self.enable_dueling_dqn
            ).to(device)

            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(
                policy_dqn.parameters(),
                lr=self.learning_rate_a
            )

            step_count = 0
            best_reward = -float("inf")

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            episode_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # ε-greedy action selection
                if is_training and random.random() < epsilon:
                    action = torch.tensor(
                        env.action_space.sample(),
                        dtype=torch.int64,
                        device=device
                    )
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax(dim=1)[0]

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                episode_reward += reward.item()

                if is_training:
                    memory.append((state, action, next_state, reward, done))
                    step_count += 1

                    if len(memory) >= self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)

                        if step_count >= self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                state = next_state

                if episode_reward >= self.stop_on_reward:
                    break

            rewards_per_episode.append(episode_reward)

            if is_training and episode_reward > best_reward:
                percent = 0.0 if best_reward == 0 else \
                    (episode_reward - best_reward) / abs(best_reward) * 100

                log_message = (
                    f"{datetime.now().strftime(DATE_FORMAT)}: "
                    f"New best reward {episode_reward:.1f} "
                    f"({percent:+.1f}%) at episode {episode}"
                )

                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            if is_training:
                now = datetime.now()
                if now - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = now

    def save_graph(self, rewards, epsilons):
        fig = plt.figure(figsize=(10, 4))

        mean_rewards = [
            np.mean(rewards[max(0, i - 99):i + 1])
            for i in range(len(rewards))
        ]

        plt.subplot(1, 2, 1)
        plt.ylabel("Mean Reward (100 eps)")
        plt.plot(mean_rewards)

        plt.subplot(1, 2, 2)
        plt.ylabel("Epsilon")
        plt.plot(epsilons)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(next_states).argmax(dim=1)
                next_q = target_dqn(next_states).gather(
                    1, best_actions.unsqueeze(1)
                ).squeeze()
            else:
                next_q = target_dqn(next_states).max(dim=1)[0]

            target_q = rewards + (1 - dones) * self.discount_factor_g * next_q

        current_q = policy_dqn(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test DQN.')
    parser.add_argument('hyperparameters', help='Hyperparameter set name')
    parser.add_argument('--train', action='store_true', help='Training mode')
    args = parser.parse_args()

    agent = Agent(args.hyperparameters)

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)
