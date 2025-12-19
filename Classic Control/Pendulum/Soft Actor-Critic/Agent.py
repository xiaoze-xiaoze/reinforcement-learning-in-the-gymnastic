import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.mean = nn.Linear(64, action_dim)
        
        self.log_std = nn.Linear(64, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t) * self.max_action
        log_prob = normal.log_prob(x_t) - torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1网络
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.FloatTensor(batch.state)
        actions = torch.FloatTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.BoolTensor(batch.done)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, learning_rate, gamma, buffer_size, batch_size, tau):
        self.action_dim = action_dim
        self.max_action = max_action
        
        # 网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        
        # 同步目标网络
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 经验池
        self.buffer = ReplayBuffer(buffer_size)
        
        # 超参数
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau      # 软更新参数
        self.alpha = 0.2    # 熵权重
        
        # 训练计数
        self.step_count = 0

    def act(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if training:
            action, _ = self.actor.sample(state)
        else:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
        return action.cpu().data.numpy().flatten()

    def remember(self, *args):
        self.buffer.push(*args)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device).unsqueeze(1)
        next_states = next_states.to(device)
        dones = dones.to(device).unsqueeze(1)

        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions) 
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q

        current_q1 = self.critic1(states, actions) 
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item() + actor_loss.item()

    def train(self, env, config):
        print("开始训练...")
        
        # 训练记录
        score = []
        eval_score = []
        losses = []
        best_score = -float('inf')
        
        for episode in range(config.train_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            episode_losses = []

            while not done:
                action = self.act(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.remember(state, action, reward, next_state, done)
                loss = self.update()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1} : Reward = {total_reward:.2f}")
            losses.append(np.mean(episode_losses) if episode_losses else 0)
            score.append(total_reward)
            
            # 每10个episode评估一次
            if (episode + 1) % 10 == 0:
                avg_eval_score = self.evaluate(env, episodes=10)
                eval_score.append(avg_eval_score)
                print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score:.2f}\n")

                # 保存最佳模型
                if avg_eval_score > best_score:
                    best_score = avg_eval_score
                    self.save_model(config.model_filename)
                    print('保存模型\n')

        print('训练完成\n')
        
        # 绘制训练曲线
        self.plot(score, losses)
        
        return score, eval_score, losses

    def evaluate(self, env, episodes=10):
        eval_scores = []
        for eval_episode in range(episodes): 
            eval_state, _ = env.reset()
            eval_reward = 0
            eval_done = False
            
            while not eval_done:
                eval_action = self.act(eval_state, training=False)
                eval_state, reward, terminated, truncated, _ = env.step(eval_action)
                eval_done = terminated or truncated
                eval_reward += reward
            
            eval_scores.append(eval_reward)
        
        return np.mean(eval_scores)

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    def test(self, env, episodes=10, render=True):
        print('开始测试...')
        
        test_scores = []
        
        for test_episode in range(episodes):
            test_state, _ = env.reset()
            test_reward = 0
            test_done = False

            while not test_done:
                test_action = self.act(test_state, training=False)
                test_state, reward, terminated, truncated, _ = env.step(test_action)
                test_done = terminated or truncated
                test_reward += reward

            test_scores.append(test_reward)
            print(f"Test Episode {test_episode + 1} : Score = {test_reward:.2f}")

        avg_score = np.mean(test_scores)
        print(f"\nAverage Test Score = {avg_score:.2f}")
        print('测试完成\n')
        
        return test_scores, avg_score

    def plot(self, scores, losses):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Training Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        plt.subplot(1, 2, 2)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()
