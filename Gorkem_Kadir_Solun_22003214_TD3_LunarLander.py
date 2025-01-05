# Görkem Kadir Solun - 22003214
# CS461 - Project 3 - TD3 Lunar Lander


"""
Building a TD3 Agent for Continuous Control
In this project, you will be implementing and evaluating the Twin Delayed Deep Deterministic policy gradient (TD3) algorithm, a state-of-the-art algorithm for solving continuous action space problems in reinforcement learning.
Background
TD3 is an algorithm that builds upon the DDPG (Deep Deterministic Policy Gradient) framework by addressing its overestimation bias. TD3 does this by using a pair of critics to reduce
value overestimation and delaying policy updates to reduce per-update error. For a detailed explanation of the TD3 algorithm, please refer to the original paper by Scott Fujimoto, Herke van Hoof, and David Meger: TD3: Twin Delayed DDPG.
Task
Your task is to implement the TD3 algorithm and test your agent in the Lunar Lander Continuous environment provided by Gymnasium. You will run your experiments using six different
random seeds and average the results. For more information on the Lunar Lander Continuous environment, please refer to the following link: Lunar Lander Continuous Environment Documentation.
Why Multiple Random Seeds?
Using multiple random seeds for the initial conditions and the environment’s stochastic elements allows us to assess the robustness and reliability of the algorithm. Averaging the results helps
in mitigating the effects of outlier performances due to particularly fortunate or unfortunate initializations, leading to more generalizable performance metrics.
Requirements
You are allowed to use machine learning libraries such as PyTorch or TensorFlow for the neural network components of TD3.
Make sure to disable rendering during training as it significantly increases the runtime.
Document your code appropriately and include instructions on how to run your implementation.
Evaluation
Your implementation will be evaluated based on the following criteria:
Correctness of Implementation: Your code should correctly implement the TD3 algorithm as described in the provided resources.
Code Clarity and Documentation: Your code should be well-organized and properly documented, making it easy to understand your implementation approach.
Performance: Your agent should achieve a minimum average reward of 200 per episode in the Lunar Lander Continuous environment to be considered successful. 
This performance should be consistent across multiple random seeds to demonstrate the robustness of your solution.
Please be aware that reaching this level of performance is essential for the assessment phase of this project. The average reward is to be calculated over a minimum of 100 consecutive
episodes to ensure statistical significance. It is not sufficient to reach this threshold in a single episode or to average this performance over the initial phase of learning; your agent must demonstrate the ability to sustain this level of performance.
Reporting Your Results
Detailed graphs or tables that show the reward per episode over time for each of the random seeds tested.
Provide an analysis of the results, discussing the stability and reliability of the learned policy.
"""

import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===========================
#   Replay Buffer Class (for storing past experiences)
# ===========================
class ReplayBuffer:
    def __init__(self, max_size=int(1e6 + 1)):
        # Initialize a ReplayBuffer object with a maximum size of max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # Add a new experience to the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        # Sample a batch from the replay buffer
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ===========================
#   Neural Network Models (Actor and Critic)
# ===========================
class Actor(nn.Module):
    """
    Policy/Actor network: state -> action
    """

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return self.max_action * torch.tanh(x)


class Critic(nn.Module):
    """
    Critic network: state-action -> Q-value
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        # Q1 forward
        x1 = torch.relu(self.l1(torch.cat([state, action], 1)))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)

        # Q2 forward
        x2 = torch.relu(self.l4(torch.cat([state, action], 1)))
        x2 = torch.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, state, action):
        """
        Returns the first Q value for TD3's "twin" part
        """
        x1 = torch.relu(self.l1(torch.cat([state, action], 1)))
        x1 = torch.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


# ===========================
#   TD3 Agent Class (Actor and Critic)
# ===========================
class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr_actor=1e-3,
        lr_critic=1e-3,
    ):
        self.gamma = gamma  # Discount factor
        self.tau = tau  # Soft update factor
        self.policy_noise = policy_noise  # Noise added to target policy
        self.noise_clip = noise_clip  # Noise clipping range
        self.policy_freq = policy_freq  # Delayed policy updates
        self.total_it = 0  # Total iterations

        # Actor and Critics
        self.actor = Actor(state_dim, action_dim, max_action)  # Local network
        self.actor_target = Actor(state_dim, action_dim, max_action)  # Target network
        self.critic = Critic(state_dim, action_dim)  # Local network
        self.critic_target = Critic(state_dim, action_dim)  # Target network

        # Copy weights from local to target
        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )  # Copy weights from local to target
        self.critic_target.load_state_dict(self.critic.state_dict())  # Copy weights

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)  # Actor
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )  # Critic

        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        """
        Selects an action using the current policy with optional Gaussian noise for exploration.
        """
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(
            0
        )  # Add batch dimension
        action = (
            self.actor(state_tensor).detach().cpu().numpy()[0]
        )  # Convert to numpy array
        # Add noise to the action for exploration
        if noise != 0:
            # Add noise to the action and clip it to the valid range
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1  # Increment total iterations

        # Sample a batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select next action according to target policy
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )  # Clip noise to be within valid range
            next_action = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )  # Clip action to be within valid range

            # Compute target Q-values using target critics
            target_Q1, target_Q2 = self.critic_target(
                next_states, next_action
            )  # Use the target networks for stability
            target_Q = torch.min(
                target_Q1, target_Q2
            )  # Use the minimum of the two Q-values (Double Q-Learning)
            target_Q = (
                rewards + (1 - dones) * self.gamma * target_Q
            )  # Bellman backup formula

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)  # Use the local networks

        # Critic loss using both Q-values
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(
            current_Q2, target_Q
        )

        # Optimize the critic (local network)
        self.critic_optimizer.zero_grad()  # Zero gradients
        critic_loss.backward()  # Compute gradients
        self.critic_optimizer.step()  # Update weights

        # Delayed policy updates (TD3 trick)
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_loss = -self.critic.Q1(
                states, self.actor(states)
            ).mean()  # Maximize Q-value

            # Optimize the actor (local network)
            self.actor_optimizer.zero_grad()  # Zero gradients
            actor_loss.backward()  # Compute gradients
            self.actor_optimizer.step()  # Update weights

            # Soft update the target networks
            for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters(),  # Update the critic target network
            ):
                target_param.data.copy_(
                    self.tau * param.data
                    + (1 - self.tau) * target_param.data  # Soft update
                )

            # Soft update the target networks (actor)
            for param, target_param in zip(
                self.actor.parameters(),
                self.actor_target.parameters(),  # Update the actor target network
            ):
                # Soft update the actor target network
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )


# ===========================
#   Training Loop
# ===========================
def run_td3_training(
    env_name="LunarLanderContinuous-v3",
    seed=0,
    max_steps=1000,
    start_timesteps=10000,  # Explore vs Exploit
    expl_noise=0.1,
    batch_size=100,
    runs_without_improvement=3,
):
    """
    Trains a TD3 agent in the given Gym environment for a specified
    number of episodes/steps. Returns a list of episode rewards.

    Args:
    - env_name: str, the Gym environment name
    - seed: int, random seed
    - max_steps: int, maximum number of steps per episode
    - start_timesteps: int, number of timesteps to collect random actions
    - expl_noise: float, Gaussian noise added to the action for exploration
    - batch_size: int, number of samples to train on each step

    Returns:
    - episode_rewards: list of floats, rewards obtained in each episode
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Create environment
    env = gym.make(env_name)
    env.action_space.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create replay buffer and agent
    replay_buffer = ReplayBuffer()
    agent = TD3Agent(state_dim, action_dim, max_action)

    episode_rewards = []
    total_steps = 0
    episode = 0
    best_avg_reward = -np.inf
    avg_reward = -np.inf
    runs_without_improvement_counter = 0

    while True:
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Select action
            if total_steps < start_timesteps:
                # Random action until start_timesteps
                action = env.action_space.sample()
            else:
                # Use policy (with some noise for exploration)
                action = agent.select_action(state, noise=expl_noise)

            # Step in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            done_bool = float(done or truncated)

            # Add to replay buffer
            replay_buffer.add(state, action, reward, next_state, done_bool)
            state = next_state
            episode_reward += reward
            total_steps += 1

            # Train agent after collecting sufficient data
            if total_steps >= start_timesteps:
                agent.train(replay_buffer, batch_size)

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        print(f"[Seed {seed}] Episode {episode+1} Reward: {episode_reward:.2f}")

        # Check for termination condition if it converges
        if len(episode_rewards) % 20 == 0 and len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                runs_without_improvement_counter = 0
                print(f"Best average reward: {best_avg_reward:.2f}")
            else:
                runs_without_improvement_counter += 1

            if runs_without_improvement_counter >= runs_without_improvement:
                print(
                    f"Training terminated due to no improvement in {runs_without_improvement} runs."
                )
                break

        episode += 1

    # Plot the learning curve with enhancements
    plt.figure(figsize=(12, 8))
    plt.plot(episode_rewards, label=f"Episode Reward of Seed {seed}", linewidth=2)
    plt.title(f"TD3 on {env_name}", fontsize=18)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", fontsize=14)
    plt.tight_layout()
    plt.show()

    env.close()
    return episode_rewards


if __name__ == "__main__":
    """
    Example usage:
      - Runs different random seeds.
      - Aggregates or plots results afterwards.
    """

    seeds = [0, 1, 2, 3, 4, 5]
    all_rewards = []

    for s in seeds:
        rewards = run_td3_training(
            env_name="LunarLanderContinuous-v3",
            seed=s,
            max_steps=1000,  # Environment steps per episode
            start_timesteps=10000,  # Use random policy for initial steps to fill buffer, explore more than exploit
            expl_noise=0.1,  # Noise for exploration
            batch_size=100,  # Batch size for training
            runs_without_improvement=3,  # Terminate training if no improvement
        )
        all_rewards.append(rewards)

    # Determine the maximum length among all rewards lists
    max_length = max(len(rewards) for rewards in all_rewards)

    # Pad shorter lists with the mean of their last 100 values
    def pad_with_mean_last_100(rewards, max_length):
        if len(rewards) >= 100:
            pad_value = np.mean(rewards[-100:])
        else:
            pad_value = np.mean(rewards)
        return rewards + [pad_value] * (max_length - len(rewards))

    all_rewards_padded = [
        pad_with_mean_last_100(rewards, max_length) for rewards in all_rewards
    ]

    # Convert to a NumPy array
    all_rewards_np = np.array(all_rewards_padded)

    # Compute average and standard deviation
    avg_rewards = np.mean(all_rewards_np, axis=0)
    std_rewards = np.std(all_rewards_np, axis=0)

    # Save the results
    np.save("td3_lunarlander_rewards.npy", all_rewards_np)

    # Plot the learning curve
    plt.figure(figsize=(16, 10))
    plt.plot(
        avg_rewards,
        label=f"Average Reward across {len(seeds)} seeds",
        linewidth=2,
        color="b",
    )
    plt.fill_between(
        range(len(avg_rewards)),
        avg_rewards - std_rewards,
        avg_rewards + std_rewards,
        alpha=0.3,
        color="b",
        label="Std Dev",
    )
    plt.title("TD3 on LunarLanderContinuous-v3", fontsize=20)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", fontsize=16)
    plt.tight_layout()
    plt.show()
