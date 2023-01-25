import numpy as np

from PPO1 import PPO
from Environment import Environment
import utils
import torch
import pickle

#################################### Testing ###################################


def test():

    print("============================================================================================")

    ####### Environment Parameters ############
    NumOfAgents = 2
    NumOfTargets = 1
    MapDimensions = (30, 30)
    AreaSize = (1000, 1000)
    AgentWindow = (7, 7)
    Idle = 1
    NumOfActions = 8 + Idle
    NumOfWalls = 1
    Speed = 50
    n_observations = 8
    DistanceThreshold = 50
    max_ep_len = 100  # max timesteps in one episode
    WallsDataset = pickle.load(open("../Datasets/Dataset_{}walls.pkl".format(NumOfWalls), 'rb'))

    env = Environment(NumOfAgents, NumOfTargets, MapDimensions, NumOfActions, NumOfWalls, AreaSize, AgentWindow, Speed,
                      DistanceThreshold, Idle, n_observations, max_ep_len, WallsDataset)

    env_name = "MultiAgentLocal"

    total_test_episodes = 100    # total num of testing episodes

    K_epochs = 20               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lmbda = 0.95

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    Learn = 0  # we are testing

    n_workers = 1
    n_steps = 4000
    EncoderLocation = "../Embeddings/Encoder{}.pt".format(NumOfWalls)
    EncoderDims = [128, 256]

    #####################################################

    # state space dimension
    state_dim = env.observation_space_size

    # action space dimension
    action_dim = env.NumOfActions

    Expert_Path = "Expert{}.pt".format(NumOfAgents)
    ExpertRate = 0
    Expert = 0
    ExpertThresh = 0.5 * 1 / NumOfActions

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, MapDimensions, action_dim, lr_actor, lr_critic, gamma, lmbda, K_epochs, eps_clip,
                    n_workers, n_steps, n_observations, NumOfAgents, EncoderLocation, EncoderDims, Expert_Path,
                    ExpertRate)

    directory = "PPO_preTrained" + '/' + "MultiAgentLocal" + '/'

    fileName = "PPO_{}_{}_{}_{}_{}_{}_{}".format("MultiAgentLocal", NumOfAgents, NumOfActions,
                                                 MapDimensions, AgentWindow, env.TargetsStrengths, NumOfWalls)
    checkpoint_path = directory + fileName + ".pt"
    print("loading network from : " + checkpoint_path + "_actor.pth")

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    TotalTime = 0
    TotalCost = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        InvalidActions = np.zeros((n_workers, NumOfAgents, NumOfActions))
        obs = np.zeros((n_workers, NumOfAgents, n_observations, state_dim[0], state_dim[1]))

        obs[0], InvalidActions[0], Walls = env.reset()
        Walls = np.expand_dims(Walls, 0)

        path = []
        pathGrid = []
        path.append(env.AgentsLocations)
        pathGrid.append(env.AgentsLocationsGrid)
        done = 1
        cost = 0
        LocTime = 0
        for t in range(1, max_ep_len+1):
            actions, _ = ppo_agent.select_action(obs, Walls, InvalidActions, Learn, t, done, Expert, ExpertThresh)

            for i in range(NumOfAgents):
                if actions[0][i] != NumOfActions - 1:
                    cost += 1

            LocTime += 1

            obs[0], reward, done, InvalidActions[0] = env.step(actions[0])
            ep_reward += reward
            path.append(env.AgentsLocations)

            if done:
                break

        TotalTime += LocTime
        TotalCost += cost
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {} \t\t Cost: {} \t\t Time: {}'.format(ep, round(ep_reward, 2), cost, LocTime))
        WallEndPoints = []
        for wall in range(NumOfWalls):
            WallEndPoints.append([env.WallsPoints[wall][0], env.WallsPoints[wall][-1]])

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("Average Time : {}".format(TotalTime/total_test_episodes))
    print("Average Cost : {}".format(TotalCost/total_test_episodes))

    print("============================================================================================")


if __name__ == '__main__':

    test()
