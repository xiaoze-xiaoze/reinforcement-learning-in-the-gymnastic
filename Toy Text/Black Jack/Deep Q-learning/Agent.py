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

# 网络结构
class QNet(nn.Module):
    def __init__(self, state_action, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_action, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_dim)

    # 前向传播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
# 经验回放
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)    # 经验回放缓冲区

    # 交互push进来
    def push(self, *args):
        self.buffer.append(Transition(*args))

    # 随机采集一个batch
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)    # 随机采样
        batch = Transition(*zip(*transitions))    # 拆包
        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.BoolTensor(batch.done)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, buffer_size, batch_size, target_sync):
        # 网络
        self.q_net = QNet(state_dim, action_dim).to(device)
        self.target_net = QNet(state_dim, action_dim).to(device)
        # 同步主网络和影子网络
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()    # 影子网络只推理不求梯度

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # 经验池
        self.buffer = ReplayBuffer(buffer_size)

        # 探索和训练计数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.step_count = 0

        # 超参数
        self.action_dim = action_dim    
        self.gamma = gamma    # 折扣因子
        self.batch_size = batch_size
        self.target_sync = target_sync    # 目标网络同步频率


    def act(self, state):
        # ε-贪婪策略
        if random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # 贪婪动作
        state = torch.FloatTensor(state).unsqueeze(0).to(device)    # unsqueeze(0)增加一个维度
        with torch.no_grad():
            q_values = self.q_net(state)    # 计算Q值
        return q_values.argmax().item()    # 返回动作索引
        
    def remember(self, *args):
        self.buffer.push(*args)

    def update(self):
        # 经验回放
        if len(self.buffer) < self.batch_size:
            return
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.float().to(device)
        actions = actions.long().to(device)
        rewards = rewards.float().to(device)
        next_states = next_states.float().to(device)
        dones = dones.bool().to(device)

        # 当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))

        # 目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (~dones))    # ~dones:布尔取反

        # 计算损失和更新
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 同步目标网络
        self.step_count += 1
        if self.step_count % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def train(self, env, config):
        print("开始训练...")
        
        # 训练记录
        score = []
        eval_score = []
        losses = []
        best_score = -1.0  # Black Jack的最低分数
        
        for episode in range(config.train_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            episode_losses = []

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
                avg_eval_score = self.evaluate(env, episodes=100)
                eval_score.append(avg_eval_score)
                print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score:.2f}\n")

                # 保存最佳模型
                if avg_eval_score > best_score + 0.01:
                    best_score = avg_eval_score
                    self.save_model(config.model_filename)

        print('训练完成\n')
        
        # 绘制训练曲线
        self.plot(score, losses)
        
        return score, eval_score, losses

    def evaluate(self, env, episodes=100):
        # 保存当前epsilon和训练状态
        eval_epsilon = self.epsilon
        self.epsilon = 0.0  # 纯贪婪策略
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
        
        # 恢复训练状态
        self.epsilon = eval_epsilon
        self.q_net.train()
        
        return np.mean(eval_scores)

    def save_model(self, model_path):
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.q_net.state_dict(), model_path)

    def load_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))

    def test(self, env, episodes=10, render=True):
        print('开始测试...')
        
        # 设置为纯贪婪策略
        self.epsilon = 0.0
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
                
                if render:
                    import time
                    time.sleep(2)

            test_scores.append(test_reward)
            print(f"Test Episode {test_episode + 1} : Score = {test_reward}")

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