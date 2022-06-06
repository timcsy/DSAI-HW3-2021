import numpy as np
from dqn_model import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import collections

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
SYNC_TARGET = 1000
LEARNING_RATE = 1e-4
REPLAY_START_SIZE = 32

EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

device = 'cpu'
net = DQN().to(device)
tgt_net = DQN().to(device)
exp_buffer = ExperienceBuffer(REPLAY_SIZE)
epsilon = EPSILON_START
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
idx = 0

def train(state, action, state_n):
    exp = Experience(state, action, reward, False, state_n)
    exp_buffer.append(exp)
    if len(exp_buffer)>=REPLAY_START_SIZE:
        if idx % SYNC_TARGET == 0:
           tgt_net.load_state_dict(net.state_dict())
        optimizer.zero_grad()
        loss_t = calc_loss()
        loss_t.backward()
        optimizer.step()
        idx += 1

def calc_loss():
    states, actions, rewards, dones, next_states = exp_buffer.sample(BATCH_SIZE)

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v.float()).gather( 1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v.float()).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

def reward():
    pass