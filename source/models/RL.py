from models import DQN, DDPG, AC
import numpy as np
import matplotlib.pyplot as plt
from function import *


def dqn():
    size = 1000
    kline_data = np.array(get_klines_data(size)).astype('float64')
    data = np.delete(kline_data, [0, 6, 11], axis=1)
    env = DQN.Environment(data)
    agent = DQN.Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=3, eps_end=0.1, n_state=5, lr=0.003)
    model_profits = []
    n_games = 50  # 训练局数

    for i in range(n_games):
        profits = [0]
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, op = env.step(action)
            profits.append(reward)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        print('episode', i,
              'epsilon %.2f' % agent.epsilon,
              'profits %.2f' % sum(profits))
        model_profits.append(sum(profits))

    fund, buy, sell = [], [], []
    profit = 0
    profits = [0]
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, op = env.step(action)
        profits.append(reward)
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_

        if op == 1:
            buy.append(env.barpos)
        elif op == 2:
            sell.append(env.barpos)
        profit += reward
        fund.append(profit)

    print('episode', n_games,
          'epsilon %.2f' % agent.epsilon,
          'profits %.2f' % sum(profits))
    model_profits.append(sum(profits))

    # plt.plot(range(len(fund)), fund)
    # plt.show()
    plt.figure(figsize=(16, 8))
    plt.plot(env.data['close'].iloc[1:], linewidth=1.2)
    plt.plot(env.data['close'].iloc[1:], '^', markersize=10, label='buying signal', markevery=buy)
    plt.plot(env.data['close'].iloc[1:], 'v', markersize=10, label='selling signal', markevery=sell)
    plt.legend(loc='upper right')
    plt.show()


def ddpg():
    size = 1000
    kline_data = np.array(get_klines_data(size)).astype('float64')
    data = np.delete(kline_data, [0, 6, 11], axis=1)
    env = DDPG.Environment(data)
    agent = DDPG.Agent(gamma=0.99, tau=0.001, batch_size=32, n_actions=3,
                       input_dims=5, lr_actor=0.0001, lr_critic=0.001)
    model_profits = []
    n_games = 100  # 训练局数

    for i in range(n_games):
        profits = [0]
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, op = env.step(action)
            profits.append(reward)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        print('episode', i,
              'epsilon %.2f' % agent.tau,
              'profits %.2f' % sum(profits))
        model_profits.append(sum(profits))

    fund, buy, sell = [], [], []
    profit = 0
    profits = [0]
    done = False
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, op = env.step(action)
        profits.append(reward)
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_

        if op == 1:
            buy.append(env.barpos)
        elif op == 2:
            sell.append(env.barpos)
        profit += reward
        fund.append(profit)

    print('episode', n_games,
          'epsilon %.2f' % agent.tau,
          'profits %.2f' % sum(profits))
    model_profits.append(sum(profits))

    # plt.plot(range(len(fund)), fund)
    # plt.show()
    plt.figure(figsize=(16, 8))
    plt.plot(env.data['close'].iloc[1:], linewidth=1.2)
    plt.plot(env.data['close'].iloc[1:], '^', markersize=10, label='buying signal', markevery=buy)
    plt.plot(env.data['close'].iloc[1:], 'v', markersize=10, label='selling signal', markevery=sell)
    plt.legend(loc='upper right')
    plt.show()


def actor_critic():
    size = 1000
    kline_data = np.array(get_klines_data(size)).astype('float64')
    data = np.delete(kline_data, [0, 6, 11], axis=1)
    env = AC.Environment(data)
    agent = AC.ActorCritic(n_states=5, n_hiddens=16, n_actions=3,
                           actor_lr=0.003, critic_lr=0.003, gamma=0.99)
    return_list = []
    n_games = 100  # 训练局数

    for i in range(n_games):
        state = env.reset()  # 环境重置
        done = False  # 任务完成的标记
        episode_return = 0  # 累计每回合的reward

        # 构造数据集，保存每个回合的状态数据
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }
        while not done:
            action = agent.take_action(state)  # 动作选择
            next_state, reward, done, op = env.step(action)  # 环境更新
            # 保存每个时刻的状态\动作\...
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新状态
            state = next_state
            # 累计回合奖励
            episode_return += reward
        # 保存每个回合的return
        return_list.append(episode_return)
        # 模型训练
        agent.update(transition_dict)

        # 打印回合信息
        print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

    plt.plot(return_list)
    plt.title('return')
    plt.show()
