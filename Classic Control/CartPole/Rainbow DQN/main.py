import gymnasium as gym
from Agent import RainbowDQNAgent
import os

class Config:
    # 网络参数
    learning_rate = 0.0001   
    gamma = 0.99
    
    # 经验回放参数
    buffer_size = 50000   
    batch_size = 32
    
    # 目标网络同步频率
    target_update_freq = 500   
    
    # 多步学习参数
    n_step = 3   
    
    # 分布式强化学习参数
    num_atoms = 51            # 分布的原子数量
    v_min = 0                 # CartPole的最小价值
    v_max = 500               # CartPole的最大价值
    
    # 优先级经验回放参数
    alpha = 0.6               # 优先级指数
    beta = 0.4                # 重要性采样指数
    beta_increment = 0.001    # beta递增量
    
    # 训练参数
    train_episodes = 500  
    
    # 测试参数
    test_episodes = 10
    
    # 环境参数
    env_name = 'CartPole-v1'
    max_episode_steps = 2000
    
    # 模型保存路径 - 使用脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, 'model', 'rainbow_best_model.pth')

def create_agent():
    config = Config()
    
    # 创建环境
    env = gym.make(config.env_name, max_episode_steps=config.max_episode_steps)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    
    # 创建Rainbow DQN Agent
    agent = RainbowDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_update_freq=config.target_update_freq,
        n_step=config.n_step,
        num_atoms=config.num_atoms,
        v_min=config.v_min,
        v_max=config.v_max
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
    