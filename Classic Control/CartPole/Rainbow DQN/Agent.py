import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 噪声线性层，用于探索
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 权重参数：均值和标准差
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        # 偏置参数：均值和标准差
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, input):
        if self.training:
            # 训练时使用噪声
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            # 测试时不使用噪声
            return F.linear(input, self.weight_mu, self.bias_mu)

# Rainbow DQN网络，结合对决架构和分布式强化学习
class RainbowNet(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms  # 分布式RL的原子数量
        self.v_min = v_min  # 价值分布的最小值
        self.v_max = v_max  # 价值分布的最大值
        
        # 分布式强化学习支持
        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # 特征提取层（使用噪声层）
        self.feature_layer = nn.Sequential(
            NoisyLinear(state_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU()
        )
        
        # 对决架构：价值流（Value Stream）
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, num_atoms)
        )
        
        # 对决架构：优势流（Advantage Stream）
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64),
            nn.ReLU(),
            NoisyLinear(64, action_dim * num_atoms)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 特征提取
        features = self.feature_layer(x)
        
        # 价值流和优势流
        values = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantages = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)
        
        # 对决架构组合：Q = V + A - mean(A)
        q_atoms = values + advantages - advantages.mean(dim=1, keepdim=True)
        
        # 应用softmax得到概率分布
        q_dist = F.softmax(q_atoms, dim=-1)
        
        return q_dist
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# 优先级经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment  # beta递增量
        self.max_priority = 1.0
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def push(self, *args):
        transition = Transition(*args)
        
        if self.size < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        
        # 新经验使用最大优先级
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        if self.size == 0:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 处理批次数据
        batch = Transition(*zip(*samples))
        
        states = torch.FloatTensor(batch.state).to(device)
        actions = torch.LongTensor(batch.action).to(device)
        rewards = torch.FloatTensor(batch.reward).to(device)
        next_states = torch.FloatTensor(batch.next_state).to(device)
        dones = torch.BoolTensor(batch.done).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return self.size

# 经验回放的数据结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Rainbow DQN智能体
class RainbowDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0001, gamma=0.99, 
                 buffer_size=100000, batch_size=32, target_update_freq=1000, n_step=3, 
                 num_atoms=51, v_min=-10, v_max=10):
        
        # 主网络和目标网络
        self.q_net = RainbowNet(state_dim, action_dim, num_atoms, v_min, v_max).to(device)
        self.target_net = RainbowNet(state_dim, action_dim, num_atoms, v_min, v_max).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
        # 优先级经验回放
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        
        # 多步学习
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        
        # 分布式强化学习参数
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.supports = torch.linspace(v_min, v_max, num_atoms).to(device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # 训练参数
        self.step_count = 0
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_dist = self.q_net(state)
            # 计算期望Q值
            q_values = (q_dist * self.supports).sum(dim=-1)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        # 添加到n步缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # 计算n步回报
            n_step_state, n_step_action = self.n_step_buffer[0][:2]
            n_step_reward = 0
            n_step_next_state, n_step_done = self.n_step_buffer[-1][3:]
            
            # 累积折扣奖励
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    n_step_next_state = self.n_step_buffer[i][3]
                    n_step_done = True
                    break
            
            self.buffer.push(n_step_state, n_step_action, n_step_reward, n_step_next_state, n_step_done)
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        # 从优先级回放缓冲区采样
        sample_data = self.buffer.sample(self.batch_size)
        if sample_data is None:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = sample_data
        
        # 重置噪声
        self.q_net.reset_noise()
        self.target_net.reset_noise()
        
        # 当前Q分布
        current_q_dist = self.q_net(states)
        current_q_dist = current_q_dist[range(self.batch_size), actions]
        
        with torch.no_grad():
            # Double DQN：使用在线网络选择动作
            next_q_dist = self.q_net(next_states)
            next_q_values = (next_q_dist * self.supports).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=1)
            
            # 使用目标网络评估选择的动作
            target_q_dist = self.target_net(next_states)
            target_q_dist = target_q_dist[range(self.batch_size), next_actions]
            
            # 分布式贝尔曼更新
            target_support = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * self.supports.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.v_min, self.v_max)
            
            # 投影到支撑集
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # 处理边界情况
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1
            
            # 分布概率
            target_q_dist_projected = torch.zeros_like(target_q_dist)
            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).to(device)
            
            target_q_dist_projected.view(-1).index_add_(0, (l + offset).view(-1), (target_q_dist * (u.float() - b)).view(-1))
            target_q_dist_projected.view(-1).index_add_(0, (u + offset).view(-1), (target_q_dist * (b - l.float())).view(-1))
        
        # 交叉熵损失
        loss = -(target_q_dist_projected * current_q_dist.log()).sum(dim=1)
        
        # 应用重要性采样权重
        weighted_loss = (loss * weights).mean()
        
        # 更新优先级
        td_errors = loss.detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-6
        self.buffer.update_priorities(indices, new_priorities)
        
        # 优化网络
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)  # 梯度裁剪
        self.optimizer.step()
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return weighted_loss.item()
    
    def train(self, env, config):
        print("开始Rainbow DQN训练...")
        
        # 训练记录
        score = []
        eval_score = []
        losses = []
        best_score = 0
        
        for episode in range(config.train_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            episode_losses = []
            
            # 每个episode开始时重置噪声
            self.q_net.reset_noise()
            
            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.remember(state, action, reward, next_state, done)
                loss = self.update()
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                total_reward += reward
            
            print(f"Episode {episode + 1} : Reward = {total_reward}")
            losses.append(np.mean(episode_losses) if episode_losses else 0)
            score.append(total_reward)
            
            # 每10个episode评估一次
            if (episode + 1) % 10 == 0:
                avg_eval_score = self.evaluate(env, episodes=10)
                eval_score.append(avg_eval_score)
                print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score}\n")
                
                # 保存最佳模型
                if avg_eval_score > best_score + 10:
                    best_score = avg_eval_score
                    self.save_model(config.model_filename)
                    print('保存Rainbow DQN模型\n')
        
        print('Rainbow DQN训练完成\n')
        
        # 绘制训练曲线
        self.plot(score, losses)
        
        return score, eval_score, losses
    
    def evaluate(self, env, episodes=10):
        self.q_net.eval()
        eval_scores = []
        
        for eval_episode in range(episodes):
            eval_state, _ = env.reset()
            eval_reward = 0
            eval_done = False
            
            while not eval_done:
                with torch.no_grad():
                    eval_action = self.act(eval_state)
                eval_state, reward, terminated, truncated, _ = env.step(eval_action)
                eval_done = terminated or truncated
                eval_reward += reward
            
            eval_scores.append(eval_reward)
        
        self.q_net.train()
        return np.mean(eval_scores)
    
    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count
        }, model_path)
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
    
    def test(self, env, episodes=10, render=True):
        print('开始Rainbow DQN测试...')
        
        self.q_net.eval()
        test_scores = []
        
        for test_episode in range(episodes):
            test_state, _ = env.reset()
            test_reward = 0
            test_done = False
            
            while not test_done:
                test_action = self.act(test_state)
                test_state, reward, terminated, truncated, _ = env.step(test_action)
                test_done = terminated or truncated
                test_reward += reward
            
            test_scores.append(test_reward)
            print(f"Test Episode {test_episode + 1} : Score = {test_reward}")
        
        avg_score = np.mean(test_scores)
        print(f"\nAverage Test Score = {avg_score}")
        print('Rainbow DQN测试完成\n')
        
        return test_scores, avg_score
    
    def plot(self, scores, losses):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(scores)
        plt.title('Rainbow DQN训练奖励')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(losses)
        plt.title('Rainbow DQN训练损失')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()