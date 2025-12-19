import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 策略网络
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

# 价值网络
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, clip, epoch, batch_size):
        self.policy_net = PolicyNet(state_dim, action_dim).to(device)
        self.value_net = ValueNet(state_dim).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.clip = clip
        self.epoch = epoch
        self.batch_size = batch_size

        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_prob = []
        self.values = []
        self.dones = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_prob = self.policy_net(state)
            value = self.value_net(state)
        
        # 采样动作
        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def remember(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_prob.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        # 计算折扣奖励和优势
        returns = []
        advantages = []

        # 计算回报
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        states = torch.FloatTensor(self.states).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        returns = torch.FloatTensor(returns).to(device)
        old_log_prob = torch.FloatTensor(self.log_prob).to(device)
        values = torch.FloatTensor(self.values).to(device)

        # 计算优势
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(self.epoch):
            action_prob = self.policy_net(states)
            dist = Categorical(action_prob)
            new_log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            new_values = self.value_net(states).squeeze()

            ratio = torch.exp(new_log_prob - old_log_prob)

            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            # 价值损失
            value_loss = F.mse_loss(new_values, returns)

            # 更新网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # 清空缓存
        self.clear_memory()

        return policy_loss.item(), value_loss.item()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_prob = []
        self.values = []
        self.dones = []

    def train(self, env, config):
        print("开始训练...")
        
        scores = []
        policy_losses = []
        value_losses = []
        best_score = 0
        
        for episode in range(config.train_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, log_prob, value = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.remember(state, action, reward, log_prob, value, done)
                
                state = next_state
                total_reward += reward
            
            # 每个episode结束后更新
            policy_loss, value_loss = self.update()
            
            scores.append(total_reward)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            
            print(f"Episode {episode + 1} : Reward = {total_reward}")
            
            # 每10个episode评估一次
            if (episode + 1) % 10 == 0:
                avg_eval_score = self.evaluate(env, episodes=10)
                print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score}\n")
                
                # 保存最佳模型
                if avg_eval_score > best_score + 10:
                    best_score = avg_eval_score
                    self.save_model(config.model_filename)
                    print('保存模型\n')
        
        print('训练完成\n')
        
        # 绘制训练曲线
        self.plot(scores, policy_losses, value_losses)
        
        return scores, policy_losses, value_losses

    def evaluate(self, env, episodes=10):
        eval_scores = []
        
        for eval_episode in range(episodes):
            eval_state, _ = env.reset()
            eval_reward = 0
            eval_done = False
            
            while not eval_done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(eval_state).unsqueeze(0).to(device)
                    action_probs = self.policy_net(state_tensor)
                    action = torch.argmax(action_probs).item()  # 贪婪策略
                
                eval_state, reward, terminated, truncated, _ = env.step(action)
                eval_done = terminated or truncated
                eval_reward += reward
            
            eval_scores.append(eval_reward)
        
        return np.mean(eval_scores)

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict()
        }, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])

    def test(self, env, episodes=10):
        print('开始测试...')
        
        test_scores = []
        
        for test_episode in range(episodes):
            test_state, _ = env.reset()
            test_reward = 0
            test_done = False
            
            while not test_done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(device)
                    action_probs = self.policy_net(state_tensor)
                    action = torch.argmax(action_probs).item()
                
                test_state, reward, terminated, truncated, _ = env.step(action)
                test_done = terminated or truncated
                test_reward += reward
            
            test_scores.append(test_reward)
            print(f"Test Episode {test_episode + 1} : Score = {test_reward}")
        
        avg_score = np.mean(test_scores)
        print(f"\nAverage Test Score = {avg_score}")
        print('测试完成\n')
        
        return test_scores, avg_score

    def plot(self, scores, policy_losses, value_losses):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(scores)
        plt.title('Training Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 3, 2)
        plt.plot(policy_losses)
        plt.title('Policy Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(value_losses)
        plt.title('Value Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()