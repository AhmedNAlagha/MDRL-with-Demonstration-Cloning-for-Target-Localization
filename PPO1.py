import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
import random
import numpy as np
import torch.nn.functional as F
import time

# torch.manual_seed(0)


################################## set device ##################################

# print("============================================================================================")

device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    # print("Device set to : " + str(torch.cuda.get_device_name(device)))


# print("============================================================================================")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self, n_workers, n_steps, state_dim, MapDims, n_channels, n_agents, EmbedDim):
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.n_channels = n_channels
        self.n_agents = n_agents
        self.MapDims = MapDims
        self.EmbedDim = EmbedDim

        self.rewards = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.float32)
        self.actions = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.int32)
        self.is_terminals = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.bool)
        self.states = np.zeros((self.n_workers, self.n_agents, self.n_steps, self.n_channels, self.state_dim[0],
                                self.state_dim[1]), dtype=np.float32)
        self.WallEmbs = np.zeros((self.n_workers, self.n_steps, 1, self.EmbedDim),
                                 dtype=np.float32)
        self.logprobs = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.float32)

    def clear(self):
        self.rewards = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.float32)
        self.actions = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.int32)
        self.is_terminals = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.bool)
        self.states = np.zeros((self.n_workers, self.n_agents, self.n_steps, self.n_channels, self.state_dim[0],
                                self.state_dim[1]), dtype=np.float32)
        self.WallEmbs = np.zeros((self.n_workers, self.n_steps, 1, self.EmbedDim),
                                 dtype=np.float32)
        self.logprobs = np.zeros((self.n_workers, self.n_agents, self.n_steps), dtype=np.float32)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_channels, EncoderDim):
        super(Actor, self).__init__()

        self.n_channels = n_channels

        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(
            int(math.ceil(state_dim[0] / 2) * math.ceil(state_dim[1] / 2) * 64) + EncoderDim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, obs, wall_emb):
        out = F.relu(self.conv1(obs), inplace=True)
        out = self.pool(out)
        out = F.relu(self.conv2(out), inplace=True)
        out = torch.flatten(out, 1)

        out = torch.cat((out, wall_emb), 1)

        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = F.softmax(self.fc3(out), dim=-1)

        return out


class Critic(nn.Module):
    def __init__(self, state_dim, n_channels, EncoderDim):
        super(Critic, self).__init__()

        self.n_channels = n_channels

        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(
            int(math.ceil(state_dim[0] / 2) * math.ceil(state_dim[1] / 2) * 64) + EncoderDim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs, wall_emb):
        out = F.relu(self.conv1(obs), inplace=True)
        out = self.pool(out)
        out = F.relu(self.conv2(out), inplace=True)
        out = torch.flatten(out, 1)

        out = torch.cat((out, wall_emb), 1)

        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)

        out = self.fc3(out)

        return out


class ActorCritic:
    def __init__(self, state_dim, action_dim, n_agents, n_workers):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_workers = n_workers

    def act(self, state, wall_emb, InvalidActions, Learn, actor, ExptActor, Expert, ExpertThresh):
        action_probs = actor(state, wall_emb).cpu() * (1 - InvalidActions) + 1e-20
        action_probs = action_probs / torch.sum(action_probs, dim=1, keepdim=True)
        dist = Categorical(action_probs)
        CheckMax1 = torch.zeros(1)
        CheckMax2 = torch.zeros(1)

        if Learn:
            actions = dist.sample()
        else:
            actions = torch.argmax(action_probs, dim=1)

        if np.sum(Expert) > 0:
            state2 = torch.clone(state[Expert, 0:7])
            Expert_action_probs = ExptActor(state2, wall_emb[Expert]).cpu() * (1 - InvalidActions[Expert]) + 1e-20
            Expert_action_probs = Expert_action_probs / torch.sum(Expert_action_probs, dim=1, keepdim=True)
            CheckMax1 = torch.argmax(Expert_action_probs, dim=1)
            Expert_action_probs = Expert_action_probs * (action_probs[Expert] >
                                                         (torch.ones(np.sum(Expert), self.action_dim) * ExpertThresh))
            CheckMax2 = torch.argmax(Expert_action_probs, dim=1)
            if Learn:
                actions[Expert] = torch.argmax(Expert_action_probs, dim=1)

        actions_logprob = dist.log_prob(actions)

        return actions.detach(), actions_logprob.detach(), torch.sum(CheckMax1 == CheckMax2).numpy()

    def evaluate(self, state, walls, action, actor, critic):

        action_logprobs = torch.zeros(self.n_workers, self.n_agents, action.shape[2]).to(device)
        dist_entropy = torch.zeros(self.n_workers, self.n_agents, action.shape[2]).to(device)
        state_values = torch.zeros(self.n_workers, self.n_agents, action.shape[2], 1).to(device)

        for i in range(self.n_workers):
            walls_n = walls[i, :, -1, :]
            for j in range(self.n_agents):
                action_probs = actor(state[i, j, :], walls_n)
                dist = Categorical(action_probs)
                action_logprobs[i, j, :] = dist.log_prob(action[i, j])
                dist_entropy[i, j, :] = dist.entropy()
                state_values[i, j, :] = critic(state[i, j, :], walls_n)

        return action_logprobs, state_values, dist_entropy


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super(Encoder, self).__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, (3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, (3, 3), stride=(2, 2), padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, fc2_input_dim),
            nn.ReLU(True),
            nn.Linear(fc2_input_dim, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class PPO:
    def __init__(self, state_dim, MapDim, action_dim, lr_actor, lr_critic, gamma, lmbda, K_epochs, eps_clip, n_workers, n_steps,
                 n_channels, NumOfAgents, EncoderLocation, EncoderDims, ExpertLocation, ExpertRate):

        self.state_dim = state_dim
        self.n_workers = n_workers
        self.n_steps = n_steps
        self.n_channels = n_channels
        self.NumOfAgents = NumOfAgents

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer(n_workers, n_steps, state_dim, MapDim, n_channels, self.NumOfAgents,
                                    EncoderDims[0])

        self.actor = Actor(state_dim, action_dim, n_channels, EncoderDims[0]).to(device)
        self.critic = Critic(state_dim, n_channels, EncoderDims[0]).to(device)

        if ExpertRate > 0:
            self.ExpActor = Actor(state_dim, action_dim, n_channels-1, EncoderDims[0]).to(device)
            if torch.cuda.is_available():
                self.ExpActor.load_state_dict(torch.load(ExpertLocation)['actor_dict'])
            else:
                self.ExpActor.load_state_dict(torch.load(ExpertLocation, map_location=torch.device('cpu'))['actor_dict'])
        else:
            self.ExpActor = None

        self.Encoder = Encoder(EncoderDims[0], EncoderDims[1]).to(device)
        if torch.cuda.is_available():
            self.Encoder.load_state_dict(torch.load(EncoderLocation))
        else:
            self.Encoder.load_state_dict(torch.load(EncoderLocation, map_location=torch.device('cpu')))

        # self.Walls = np.zeros(self.n_workers, 1, MapDim[0], MapDim[1])
        self.wall_emb = np.zeros((self.n_workers, 1, EncoderDims[0]))

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        self.actor_old = Actor(state_dim, action_dim, n_channels, EncoderDims[0]).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old = Critic(state_dim, n_channels, EncoderDims[0]).to(device)
        self.critic_old.load_state_dict(self.critic.state_dict())

        self.MseLoss = nn.MSELoss()

        self.Policy = ActorCritic(state_dim, action_dim, NumOfAgents, n_workers)

    def select_action(self, state, Walls, InvalidActions, Learn, t, done, Expert, ExpertThresh):

        actions = np.zeros((self.n_workers, self.NumOfAgents))
        actions_log = np.zeros((self.n_workers, self.NumOfAgents))

        if np.sum(done) > 0:
            with torch.no_grad():
                Walls_n = torch.FloatTensor(Walls).to(device)
                self.wall_emb = self.Encoder(Walls_n)

        self.buffer.WallEmbs[:, t, -1, :] = self.wall_emb.cpu().numpy()

        for i in range(self.NumOfAgents):
            InvActions = InvalidActions[:, i, :]
            with torch.no_grad():
                state_n = torch.FloatTensor(state[:, i]).to(device)
                action, action_logprob, ExpertUse = self.Policy.act(state_n, self.wall_emb, InvActions, Learn, self.actor_old,
                                                         self.ExpActor, Expert, ExpertThresh)
                actions[:, i] = action
                actions_log[:, i] = action_logprob

            self.buffer.states[:, i, t, :, :, :] = state_n.cpu().numpy()
            self.buffer.actions[:, i, t] = action.numpy()
            self.buffer.logprobs[:, i, t] = action_logprob.numpy()

        return actions.astype(int), ExpertUse

    def update(self):

        old_states = torch.tensor(self.buffer.states).detach().to(device)
        old_walls = torch.tensor(self.buffer.WallEmbs).detach().to(device)
        old_actions = torch.tensor(self.buffer.actions).detach().to(device)
        old_logprobs = torch.tensor(self.buffer.logprobs).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.Policy.evaluate(old_states, old_walls, old_actions, self.actor,
                                                                        self.critic)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values, dim=-1)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            returns, advantages = self.calc_advantages(self.buffer.is_terminals, self.buffer.rewards,
                                                                  state_values)

            # returns = (returns - returns.mean()) / (returns.std() + 1e-7)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # loss = (-torch.min(surr1, surr2)).mean() + 0.5 * self.MseLoss(state_values,
            #                                                               returns) - 0.01 * dist_entropy.mean()
            loss = (-torch.min(surr1, surr2)) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save({
            'actor_dict': self.actor_old.state_dict(),
            'critic_dict': self.critic_old.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        self.actor_old.load_state_dict(checkpoint['actor_dict'])
        self.actor.load_state_dict(checkpoint['actor_dict'])
        self.critic_old.load_state_dict(checkpoint['critic_dict'])
        self.critic.load_state_dict(checkpoint['critic_dict'])

    def calc_advantages(self, is_terminal, rewards, values):
        returns = np.zeros((self.n_workers, self.NumOfAgents, self.n_steps))
        gae = np.zeros((self.n_workers, self.NumOfAgents))
        values2 = torch.cat((values, torch.unsqueeze(values[:, :, -1], dim=-1)), -1).detach().cpu().numpy()
        for k in reversed(range(self.n_steps)):
            delta = rewards[:, :, k] + self.gamma * values2[:, :, k+1] * (~is_terminal[:, :, k]) - values2[:, :, k]
            gae = delta + self.gamma * self.lmbda * (~is_terminal[:, :, k]) * gae
            returns[:, :, k] = gae + values2[:, :, k]

        adv = returns - values2[:, :, :-1]
        adv = ((adv - np.mean(adv, axis=-1, keepdims=True)) / (np.std(adv, axis=-1, keepdims=True) + 1e-10))
        return torch.tensor(returns, dtype=torch.float32).to(device), torch.tensor(adv, dtype=torch.float32).to(device)
