import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic网络
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)

        return policy, value

class A2CAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma):
        self.ac_net = ActorCriticNet(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        self.action_dim = action_dim
        self.gamma = gamma

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            policy, _ = self.ac_net(state)
        
        action = torch.multinomial(policy, 1).item()
        return action

    def update(self, states, actions, rewards, dones):
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # 计算折扣回报
        returns = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (~dones[i])
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(device)

        # 前向传播
        policy, value = self.ac_net(states)
        value = value.squeeze()

        # 计算优势
        advantage = returns - value

        # Actor损失
        log_probs = torch.log(policy.gather(1, actions.unsqueeze(1))).squeeze()
        actor_loss = -(log_probs * advantage.detach()).mean()

        # Critic损失
        critic_loss = F.mse_loss(value, returns)

        # 总损失
        total_loss = actor_loss + critic_loss

        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train(self, env, config):
        print("开始训练...")
        
        scores = []
        eval_scores = []
        losses = []
        best_score = 0
        
        for episode in range(config.train_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            # 存储一个episode的数据
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_dones = []

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_dones.append(done)

                state = next_state
                total_reward += reward

            # 更新网络
            loss = self.update(episode_states, episode_actions, episode_rewards, episode_dones)
            
            print(f"Episode {episode + 1} : Reward = {total_reward}")
            scores.append(total_reward)
            losses.append(loss)
            
            # 每10个episode评估一次
            if (episode + 1) % 10 == 0:
                avg_eval_score = self.evaluate(env, episodes=10)
                eval_scores.append(avg_eval_score)
                print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score}\n")

                # 保存最佳模型
                if avg_eval_score > best_score + 10:
                    best_score = avg_eval_score
                    self.save_model(config.model_filename)
                    print('保存模型\n')

        print('训练完成\n')
        self.plot(scores, losses)
        return scores, eval_scores, losses

    def evaluate(self, env, episodes=10):
        self.ac_net.eval()
        eval_scores = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_probs, _ = self.ac_net(state_tensor)
                    action = action_probs.argmax().item()  # 贪婪策略
                
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
            
            eval_scores.append(total_reward)
        
        self.ac_net.train()
        return np.mean(eval_scores)

    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.ac_net.state_dict(), model_path)

    def load_model(self, model_path):
        self.ac_net.load_state_dict(torch.load(model_path))

    def test(self, env, episodes=10, render=True):
        print('开始测试...')
        self.ac_net.eval()
        test_scores = []
        
        for test_episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_probs, _ = self.ac_net(state_tensor)
                    action = action_probs.argmax().item()

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

            test_scores.append(total_reward)
            print(f"Test Episode {test_episode + 1} : Score = {total_reward}")

        avg_score = np.mean(test_scores)
        print(f"\nAverage Test Score = {avg_score}")
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