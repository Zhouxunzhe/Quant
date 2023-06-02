import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import torch.nn.functional as F
from torch import optim


class Environment:
    def __init__(self, data):
        self.data = pd.DataFrame()
        self.data['open'] = data[:, 0]
        self.data['high'] = data[:, 1]
        self.data['low'] = data[:, 2]
        self.data['close'] = data[:, 3]
        self.data['pct_change'] = \
            ((self.data['close'] - self.data['close'].shift(1)) / self.data['close'].shift(1))
        self.data = self.data[1:]
        self.barpos = 1
        self.buy_fee_rate = 0.0
        self.sell_fee_rate = 0.0
        self.order_size = 100000  # 每个订单购买的价钱
        self.balance = 10000000
        self.position = 0  # 持有股份数
        self.market_value = 0
        self.fund = self.balance
        self.day_profit = 0

    def reset(self):
        self.barpos = 1
        self.balance = 10000000
        self.position = 0  # 持有股份数
        self.market_value = 0
        self.fund = self.balance
        self.day_profit = 0

        observation = list(self.data.iloc[self.barpos])
        return observation

    def step(self, action):
        op = 0  # 0:do nothing, 1:buy, 2:sell
        current_price = self.data['close'][self.barpos]
        self.day_profit = self.position * current_price * self.data['pct_change'][self.barpos]
        if np.argmax(action) == 0:
            if self.balance > self.order_size:
                buy_order = math.floor(self.order_size / self.data['close'][self.barpos])
                self.position += buy_order
                trade_amount = buy_order * current_price
                buy_fee = buy_order * current_price * self.buy_fee_rate
                self.balance = self.balance - trade_amount - buy_fee
                # print("buy:success")
                op = 1
            else:
                # print("buy:not enough fund")
                pass
        elif np.argmax(action) == 1:
            if self.position * current_price > self.order_size:
                sell_order = math.ceil(self.order_size / self.data['close'][self.barpos])
                self.position -= sell_order
                sell_fee = sell_order * current_price * self.sell_fee_rate
                trade_amount = sell_order * current_price
                self.balance = self.balance + trade_amount - sell_fee
                op = 2
                # print("sell:success")
            else:
                # print("sell:not enough stock")
                pass
        else:
            pass

        # 重新计算持仓状况
        self.market_value = self.position * current_price
        self.fund = self.market_value + self.balance
        self.barpos += 1

        observation_ = list(self.data.iloc[self.barpos])
        reward = self.day_profit
        if self.barpos == len(self.data) - 1:
            done = True
        else:
            done = False

        return observation_, reward, done, op


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_memory = np.zeros((max_size, input_shape))
        self.new_state_memory = np.zeros((max_size, input_shape))
        self.action_memory = np.zeros((max_size, n_actions))
        self.reward_memory = np.zeros(max_size)
        self.done_memory = np.zeros(max_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.ptr % self.max_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_buffer(self, batch_size):
        max_mem = min(self.size, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        done = self.done_memory[batch]

        return states, actions, rewards, states_, done


class Actor(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action

        return x


class Critic(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

    def forward(self, x, u):
        x = F.relu(self.fc1(torch.cat([x, u], 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent:
    def __init__(self, input_dims, n_actions, tau=0.001, gamma=0.99,
                 lr_actor=0.0001, lr_critic=0.001, max_mem_size=int(1e6),
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(max_mem_size, input_dims, n_actions)
        self.actor = Actor(input_dims, 256, 256, n_actions, 1.0)
        self.critic = Critic(input_dims, 256, 256, n_actions)
        self.target_actor = Actor(input_dims, 256, 256, n_actions, 1.0)
        self.target_critic = Critic(input_dims, 256, 256, n_actions)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.mse_loss = nn.MSELoss()
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float32)
        action = self.actor.forward(state)
        return action.detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.size < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        target_actions = self.target_actor.forward(states_)
        target_critic_value_ = self.target_critic.forward(states_, target_actions.detach())
        target_critic_value = rewards + self.gamma * target_critic_value_ * (1 - done)
        critic_value = self.critic.forward(states, actions)

        critic_loss = self.mse_loss(critic_value, target_critic_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()