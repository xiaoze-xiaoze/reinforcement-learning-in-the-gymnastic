import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.mean = nn.Linear(300, action_dim)
        self.log_std = nn.Linear(300, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, tau, alpha, buffer_size, batch_size):
        self.gamma = gamma
        self.tau = tau
        
        # 自适应熵调整
        self.target_entropy = -action_dim  # 目标熵 = -4 for BipedalWalker
        self.log_alpha = torch.tensor(np.log(alpha), device=device, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # 网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # 添加episode跟踪用于课程学习
        self.current_episode = 0
        
    def act(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if training:
            action, _ = self.actor.sample(state)
        else:
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        return action.cpu().data.numpy().flatten()
    
    def remember(self, state, action, reward, next_state, done):
        # 减轻摔倒惩罚
        if reward <= -100:
            shaped_reward = -1
        else:
            shaped_reward = reward
        
        self.replay_buffer.push(state, action, shaped_reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
            
        # 采样
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.BoolTensor(done).unsqueeze(1).to(device)
        
        # 更新Critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (~done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # 更新Actor
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # 自适应熵调整
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        return actor_loss.item(), (critic1_loss + critic2_loss).item()
    
    def train(self, env, config):
        print("开始训练...")
        
        scores = []
        actor_losses = []
        critic_losses = []
        best_score = -float('inf')
        total_steps = 0
        
        for episode in range(config.train_episodes):
            self.current_episode = episode
            state, _ = env.reset()
            total_reward = 0
            done = False
            episode_actor_losses = []
            episode_critic_losses = []
            
            while not done:
                action = self.act(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.remember(state, action, reward, next_state, done)
                total_steps += 1
                
                # 预热期后，每4步更新一次网络
                if total_steps > 1000 and total_steps % 4 == 0:
                    actor_loss, critic_loss = self.update()
                    episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)
                
                state = next_state
                total_reward += reward
            
            scores.append(total_reward)
            # 记录episode的平均loss
            avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0.0
            avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0.0
            actor_losses.append(avg_actor_loss)
            critic_losses.append(avg_critic_loss)
            
            print(f"Episode {episode + 1} : Reward = {total_reward:.2f}")
            
            # 每50个episode评估一次
            if (episode + 1) % 50 == 0:
                avg_eval_score = self.evaluate(env, episodes=5)
                print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score:.2f}\n")
                
                # 保存最佳模型
                if avg_eval_score > best_score:
                    best_score = avg_eval_score
                    self.save_model(config.model_filename)
                    print('保存模型\n')
        
        print('训练完成\n')
        
        # 绘制训练曲线
        self.plot(scores, actor_losses, critic_losses)
        
        return scores, actor_losses, critic_losses
    
    def evaluate(self, env, episodes=5):
        eval_scores = []
        
        for eval_episode in range(episodes):
            eval_state, _ = env.reset()
            eval_reward = 0
            eval_done = False
            
            while not eval_done:
                action = self.act(eval_state, training=False)
                eval_state, reward, terminated, truncated, _ = env.step(action)
                eval_done = terminated or truncated
                eval_reward += reward
            
            eval_scores.append(eval_reward)
        
        return np.mean(eval_scores)
    
    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha
        }, model_path)
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
    
    def test(self, env, episodes=10):
        print('开始测试...')
        
        test_scores = []
        
        for test_episode in range(episodes):
            test_state, _ = env.reset()
            test_reward = 0
            test_done = False
            
            while not test_done:
                action = self.act(test_state, training=False)
                test_state, reward, terminated, truncated, _ = env.step(action)
                test_done = terminated or truncated
                test_reward += reward
            
            test_scores.append(test_reward)
            print(f"Test Episode {test_episode + 1} : Score = {test_reward:.2f}")
        
        avg_score = np.mean(test_scores)
        print(f"\nAverage Test Score = {avg_score:.2f}")
        print('测试完成\n')
        
        return test_scores, avg_score
    
    def plot(self, scores, actor_losses, critic_losses):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(scores)
        plt.title('Training Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 3, 2)
        plt.plot(actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
