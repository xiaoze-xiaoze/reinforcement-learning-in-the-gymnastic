import gymnasium as gym
from Agent import DQNAgent
import os

class Config:
    # 网络参数
    learning_rate = 0.001
    gamma = 0.99
    
    # 经验回放参数
    buffer_size = 20000
    batch_size = 32
    
    # 目标网络同步频率
    target_sync = 100
    
    # 训练参数
    train_episodes = 500
    
    # 测试参数
    test_episodes = 10
    
    # 环境参数
    env_name = 'FrozenLake-v1'
    max_episode_steps = 100
    is_slippery = True
    
    # 模型保存路径 - 使用脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, 'model', 'best_model.pth')

def create_agent():
    config = Config()
    
    # 创建环境
    env = gym.make(config.env_name, max_episode_steps=config.max_episode_steps, is_slippery=config.is_slippery)
    state_dim = env.observation_space.n  # 16
    action_dim = env.action_space.n  # 4
    
    # 创建Agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_sync=config.target_sync
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
    test_env = gym.make(config.env_name, render_mode="human", max_episode_steps=config.max_episode_steps)
    
    # 测试
    test_scores, avg_score = agent.test(test_env, episodes=config.test_episodes)
    
    test_env.close()

    return test_scores, avg_score

if __name__ == '__main__':
    # train()
    test()