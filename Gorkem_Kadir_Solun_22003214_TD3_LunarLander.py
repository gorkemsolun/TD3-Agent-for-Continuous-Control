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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ===========================
#   Replay Buffer
# ===========================
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ===========================
#   Neural Network Models
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
#   TD3 Agent
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
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        # Actor and Critics
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # Copy weights from local to target
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        """
        Selects an action using the current policy with optional Gaussian noise for exploration.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        if noise != 0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample a batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select next action according to target policy
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute target Q-values
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update the target networks
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )


# ===========================
#   Training Loop
# ===========================
def run_td3_training(
    env_name="LunarLanderContinuous-v3",
    seed=0,
    max_episodes=300,
    max_steps=1000,
    start_timesteps=10000,
    expl_noise=0.1,
    batch_size=100,
):
    """
    Trains a TD3 agent in the given Gym environment for a specified
    number of episodes/steps. Returns a list of episode rewards.
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

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

    for episode in range(max_episodes):
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

    env.close()
    return episode_rewards


if __name__ == "__main__":
    """
    Example usage:
      - Runs 6 different random seeds.
      - Trains for 300 episodes per seed.
      - Aggregates or plots results afterwards.
    """
    import matplotlib.pyplot as plt

    seeds = [0, 1, 2, 3, 4, 5]
    all_rewards = []

    for s in seeds:
        rewards = run_td3_training(
            env_name="LunarLanderContinuous-v3",
            seed=s,
            max_episodes=300,  # Increase if you need more training time
            max_steps=1000,  # Environment steps per episode
            start_timesteps=10000,  # Use random policy for initial steps
        )
        all_rewards.append(rewards)

    # Compute average rewards across seeds at each episode index
    # (Note: each run could have variable length, so we align by episode index)
    all_rewards = np.array(all_rewards)  # shape (num_seeds, num_episodes)
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # Plot the learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(avg_rewards, label="Average Reward across 6 seeds")
    plt.fill_between(
        range(len(avg_rewards)),
        avg_rewards - std_rewards,
        avg_rewards + std_rewards,
        alpha=0.3,
        label="Std Dev",
    )
    plt.title("TD3 on LunarLanderContinuous-v3")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
