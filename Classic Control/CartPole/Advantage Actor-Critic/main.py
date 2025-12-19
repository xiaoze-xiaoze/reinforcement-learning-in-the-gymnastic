import gymnasium as gym
from Agent import A2CAgent
import os

class Config:
    # 网络参数
    learning_rate = 0.0007
    gamma = 0.99
    
    # 训练参数
    train_episodes = 1500
    
    # 测试参数
    test_episodes = 10
    
    # 环境参数
    env_name = 'CartPole-v1'
    max_episode_steps = 2000
    
    # 模型保存路径 - 使用脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, 'model', 'best_model.pth')

def create_agent():
    config = Config()
    
    # 创建环境
    env = gym.make(config.env_name, max_episode_steps=config.max_episode_steps)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    
    # 创建Agent
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma
    )
    
    return agent, env, config

def train():
    agent, env, config = create_agent()
    
    # 训练
    train_scores, eval_scores, losses = agent.train(env, config)
    
    env.close()
    
    return train_scores, eval_scores, losses

def test():
    agent, _, config = create_agent()
    
    # 加载训练好的模型
    agent.load_model(config.model_filename)

    # 创建带渲染的测试环境
    test_env = gym.make(config.env_name, render_mode="human", max_episode_steps=None)
    
    # 测试
    test_scores, avg_score = agent.test(test_env, episodes=config.test_episodes)
    
    test_env.close()

    return test_scores, avg_score

if __name__ == '__main__':
    # train()
    test()