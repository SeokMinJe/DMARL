import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    # batch_size, lists 생성 및 저장
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = [] # flag

        self.batch_size = batch_size # 5

    # generate 20-length array & random mini batch
    def generate_batches(self):
        n_states = len(self.states) # 20
        batch_start = np.arange(0, n_states, self.batch_size) # 5 interval batch_start index array
        indices = np.arange(n_states, dtype=np.int64) # give int indices array(for shuffle)
        np.random.shuffle(indices) # indices shuffle
        batches = [indices[i:i+self.batch_size] for i in batch_start] # create shuffled mini indices batch; tensor[4,5]

        # return array
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    # append lists
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    # lists initialize
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        # make checkpoint_file
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        # actor model
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims), # input_dims, 256
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims), # 256, 256
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions), # 256, n_actions
                nn.Softmax(dim=-1) # sum=1 distribution
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # learning rate=0.003
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # if GPU possible, change device to 'cuda:0'
        self.to(self.device) # send entire network to device

    # forwarding
    def forward(self, state): # input: state -> output: action distribution
        dist = self.actor(state) # softmax distribution
        # print(dist): tensor([[0.4796, 0.5204]], grad_fn=<SoftmaxBackward>)
        dist = Categorical(dist) # Creates a categorical distribution parameterized by either probs or logits
        # print(dist): Categorical(probs: torch.Size([1, 2]))
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims), # input_dims, 256
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims), # 256, 256
                nn.ReLU(),
                nn.Linear(fc2_dims, 1) # 256, 1
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # critic model
    def forward(self, state): # input: state -> output: value
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10): # horizon; the number of steps before we perform update
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    # store memory(interface between agent & memory)
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()


    # agent's action choose
    def choose_action(self, observation):
        # convert Numpy array -> torch tensor(float type) batch dimension
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        # print('state:', state): tensor([[-0.0451,  0.0392, -0.0345, -0.0099]])

        dist = self.actor(state) # action distribution
        # print('dist:', dist): Categorical(probs: torch.Size([1, 2]))
        action = dist.sample() # sample action from action distribution
        # print('action:', action): action: tensor([1])
        value = self.critic(state)  # value
        # print('value:', value): tensor([[0.0102]], grad_fn=<AddmmBackward>)

        # squeeze to get rid of batch dimension
        probs = T.squeeze(dist.log_prob(action)).item() # log probility of action, .item(): give integer
        # print('probs:', probs): probs: -0.7535027861595154
        action = T.squeeze(action).item()
        # print('action:', action): action: 1
        value = T.squeeze(value).item()
        # print('value:', value): value: -0.09682734310626984

        return action, probs, value

    # learning
    def learn(self):
        for _ in range(self.n_epochs): # make n_epoch random minibatch of same trajectory memory[20]
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.memory.generate_batches() # 20
            # print('vals_arr:' ,vals_arr)
            # print('reward_arr:', reward_arr)
            # print('dones_arr:', dones_arr)

            # new notation
            values = vals_arr

            # advantage initialize
            advantage = np.zeros(len(reward_arr), dtype=np.float32) # 20

            # to calculate advantage
            for t in range(len(reward_arr)-1): # 0~19
                discount = 1 # discount factor initialize
                a_t = 0 # advantage

            # delta = reward_arr[k] + Gamma * value_arr[k+1] * mask - value_arr[k];
            # adv = delta + Gamma * Lambda * mask * adv;

                for k in range(t, len(reward_arr)-1): # timestep 증가하면서 reward 개수만큼 0-19 ~19
                    # delta = reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k]
                    # a_t = delta + self.gamma * self.gae_lambda * (1-int(dones_arr[k])) * a_t

                    a_t += discount*(reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t # calculated advantage at timestep t, mini batch
                # print('advantage: ', advantage)

            # convert advantage to tensor(specifically cuda)
            advantage = T.tensor(advantage).to(self.actor.device)
            # print('advantage: ', advantage)
            # advantage: tensor([11.5490, 11.2040, 10.8326, 10.4747, 10.0879, 9.6513, 9.1811, 8.6868,
            #                   8.1866, 7.6286, 7.0325, 6.3915, 5.7008, 4.9714, 4.2513, 3.4280,
            #                   2.8331, 1.9501, 1.0109, 0.0000])
            # convert value to tensor
            values = T.tensor(values).to(self.actor.device)
            # print('values:' ,values)
            # values: tensor([0.0896, 0.1023, 0.1194, 0.1016, 0.0896, 0.1014, 0.1191, 0.1316, 0.1200,
            #                 0.1333, 0.1493, 0.1723, 0.2043, 0.2317, 0.2069, 0.2365, 0.0925, 0.0924,
            #                 0.0926, 0.1046], dtype=torch.float64)


            # to optimize
            for batch in batches: # tensor[4,5]
                # convert batch's each states, old_prob, actions to tensor
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # get new dist, value from network
                dist = self.actor(states) # new distribution of actor # tensor[5,2]
                critic_value = self.critic(states) # new value of critic # tensor[1,5]
                critic_value = T.squeeze(critic_value) # tensor[,5]

                new_probs = dist.log_prob(actions) # new prob from new categorical distribution 'dist'
                # print('new_probs:', new_probs)
                prob_ratio = new_probs.exp() / old_probs.exp() # convert probs to e log
                # print('prob_ratio:', prob_ratio)
                #prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = advantage[batch] * prob_ratio # r_t * A_t
                # print('weighted_probs', weighted_probs)
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean() # get mean actor loss

                # we don't have to estimate return, just use memory's value
                returns = advantage[batch] + values[batch]
                # furthermore, critic loss is MSE between returns and critic's estimate value
                critic_loss = (returns-critic_value)**2 # critic loss
                critic_loss = critic_loss.mean()

                # 0.5 is defined heuristically
                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward() # back propagate
                self.actor.optimizer.step() # optimize
                self.critic.optimizer.step()

        # memory initialize
        self.memory.clear_memory()               


