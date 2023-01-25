import os
from PPO1 import PPO
from typing import Dict, List
import numpy as np
from Environment import Environment
import multiprocessing
import multiprocessing.connection
from datetime import datetime
import pickle


class Env:
    def __init__(self, NumOfAgents, NumOfTargets, MapDimensions, NumOfActions, NumOfWalls, AreaSize, AgentWindow,
                 Speed, DistanceThreshold, Idle, n_observations, MaxEpLength, WallsDataset):
        self.env = Environment(NumOfAgents, NumOfTargets, MapDimensions, NumOfActions, NumOfWalls,
                               AreaSize, AgentWindow, Speed, DistanceThreshold, Idle, n_observations, MaxEpLength,
                               WallsDataset)
        self.rewards = []
        self.MaxEpLength = MaxEpLength
        self.Walls = []
        self.NumOfActions = NumOfActions
        self.Idle = Idle
        self.MovementCost = 0
        self.episode_info = {}
        self.Success = 1

    def step(self, action):
        obs, r, done, InvalidActions = self.env.step(action)
        self.rewards.append(r)
        self.MovementCost += np.sum(action != (self.NumOfActions-self.Idle))
        self.episode_info = {}

        if len(self.rewards) >= self.MaxEpLength:
            done = 1
            self.Success = 0

        if done:
            self.episode_info = {"reward": sum(self.rewards), "length": len(self.rewards), "Movement Cost":
                                 self.MovementCost, "Success": self.Success}
            obs, InvalidActions, self.Walls = self.reset()

        return obs, self.Walls, r, done, self.episode_info, InvalidActions

    def reset(self):
        self.rewards = []
        self.MovementCost = 0
        self.Success = 1
        obs, InvalidActions, self.Walls = self.env.reset()
        return obs, InvalidActions, self.Walls


def worker_process(remote: multiprocessing.connection.Connection, NumOfAgents, NumOfTargets,
                   MapDimensions, NumOfActions, NumOfWalls, AreaSize, AgentWindow, Speed, DistanceThreshold, Idle,
                   n_observations, MaxEpLength, WallsDataset):

    EnvCopy = Env(NumOfAgents, NumOfTargets, MapDimensions, NumOfActions, NumOfWalls, AreaSize,
                  AgentWindow, Speed, DistanceThreshold, Idle, n_observations, MaxEpLength, WallsDataset)

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(EnvCopy.step(data))
        elif cmd == "reset":
            remote.send(EnvCopy.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    def __init__(self, NumOfAgents, NumOfTargets, MapDimensions, NumOfActions, NumOfWalls, AreaSize, AgentWindow, Speed,
                 DistanceThreshold, Idle, n_observations, MaxEpLength, WallsDataset):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process,
                                               args=(parent, NumOfAgents, NumOfTargets, MapDimensions,
                                                     NumOfActions, NumOfWalls, AreaSize, AgentWindow, Speed,
                                                     DistanceThreshold, Idle, n_observations, MaxEpLength, WallsDataset))
        self.process.start()


class Main:

    def __init__(self):

        ############################## Define Parameters ##############################

        self.NumOfAgents = 2
        self.NumOfTargets = 1
        self.MapDimensions = (30, 30)
        self.AreaSize = (1000, 1000)
        self.AgentWindow = (7, 7)
        self.NumOfWalls = 1
        self.WallsDataset = pickle.load(open("../Walls/Walls_{}.pkl".format(self.NumOfWalls), 'rb'))
        self.n_observations = 8  # if embeddings then + 1
        self.Idle = 1  # whether idle is an option or not
        self.NumOfActions = 8 + self.Idle  # better be 1+n, where n is divisible by 4 (works with any n anyways).
        self.Speed = 50
        self.MinSpeed = 20
        self.SpeedDecay = 0
        self.SpeedDecayFreq = int(1.5e6)
        self.DistanceThreshold = 50
        self.max_ep_len = 100  # max timesteps in one episode
        self.n_workers = 1

        self.training_seed = 1
        self.lr_actor = 0.0003  # learning rate for actor network
        self.lr_critic = 0.001  # learning rate for critic network
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95
        self.n_steps = int(1e8)
        self.batch_size = 4000
        self.updates = self.n_steps // self.batch_size

        self.worker_steps = self.batch_size//self.n_workers
        self.epochs = 20
        self.n_mini_batch = 4
        self.Learn = 1  # learning or testing
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        #assert (self.batch_size % self.n_mini_batch == 0)

        ############################## Define Environments/Workers ##############################

        self.workers = [Worker(self.NumOfAgents, self.NumOfTargets, self.MapDimensions,
                               self.NumOfActions, self.NumOfWalls, self.AreaSize, self.AgentWindow, self.Speed,
                               self.DistanceThreshold, self.Idle, self.n_observations, self.max_ep_len, self.WallsDataset) for _ in
                        range(self.n_workers)]
        TempEnv = Environment(self.NumOfAgents, self.NumOfTargets, self.MapDimensions,
                              self.NumOfActions, self.NumOfWalls, self.AreaSize, self.AgentWindow, self.Speed,
                              self.DistanceThreshold, self.Idle, self.n_observations, self.max_ep_len, self.WallsDataset)

        self.state_dim = TempEnv.observation_space_size
        self.action_dim = TempEnv.NumOfActions

        self.obs = np.zeros((self.n_workers, self.NumOfAgents, self.n_observations, self.state_dim[0],
                             self.state_dim[1]))
        self.Walls = np.zeros((self.n_workers, 1, self.MapDimensions[0], self.MapDimensions[1]))
        self.InvalidActions = np.zeros((self.n_workers, self.NumOfAgents, self.NumOfActions))
        self.done = np.ones(self.n_workers, dtype=int)

        for worker in self.workers:
            worker.child.send(("reset", None))

        for i, worker in enumerate(self.workers):
            self.obs[i], self.InvalidActions[i], self.Walls[i] = worker.child.recv()

        ############################## Encoder ##############################

        self.EncoderLocation = "../Embeddings/Encoder{}.pt".format(self.NumOfWalls)
        self.EncoderDims = [128, 256]

        self.PPO = PPO(self.state_dim, self.MapDimensions, self.action_dim, self.lr_actor, self.lr_critic, self.gamma,
                       self.lmbda, self.epochs, self.eps_clip, self.n_workers, self.worker_steps, self.n_observations,
                       self.NumOfAgents, self.EncoderLocation, self.EncoderDims)

        self.save_freq = 20
        self.print_freq = 5
        self.log_freq = 5
        self.save_console_freq = 5
        self.RewardCheck = 500

        ############################## Logging/Saving/Printing ##############################

        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + "MultiAgentLocal" + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = log_dir + '/PPO_' + "MultiAgentLocal" + "_log_" + str(run_num) + ".csv"

        self.directory = "PPO_preTrained"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.directory = self.directory + '/' + "MultiAgentLocal" + '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.fileName = "PPO_{}_{}_{}_{}_{}_{}_{}".format("MultiAgentLocal", self.NumOfAgents, self.NumOfActions,
                                                       self.MapDimensions, self.AgentWindow, TempEnv.TargetsStrengths,
                                                       self.NumOfWalls)
        self.checkpoint_path = self.directory + self.fileName + ".pt"

        # track total training time
        self.start_time = datetime.now().replace(microsecond=0)

        # logging file
        self.log_f = open(log_f_name, "w+")
        self.log_f.write('episode,timestep,reward\n')

        self.ConsoleOutput = ["\ncurrent logging run number for " + "MultiAgentLocal" + " : {}".format(run_num),
                              "logging at : {}".format(log_f_name),
                              "save checkpoint path : {}".format(self.checkpoint_path),
                              "---------------------------------------------------------------------------------------",
                              "max training timesteps : {}".format(self.n_steps),
                              "max timesteps per episode : {}".format(self.max_ep_len),
                              "Number of Workers : {}".format(self.n_workers),
                              "model saving frequency : every {} batches".format(self.save_freq),
                              "log frequency : every {} batches".format(self.log_freq),
                              "printing average reward over episodes in last : {} timesteps".format(self.print_freq),
                              "---------------------------------------------------------------------------------------",
                              "state space dimension : {}".format(self.state_dim),
                              "action space dimension : {}".format(self.action_dim),
                              "---------------------------------------------------------------------------------------",
                              "PPO update frequency : {} timesteps".format(self.batch_size),
                              "PPO K epochs : {}".format(self.epochs),
                              "PPO epsilong clip : {}".format(self.eps_clip),
                              "discount factor (gamma) : {}".format(self.gamma),
                              "Lambda : {}".format(self.lmbda),
                              "---------------------------------------------------------------------------------------",
                              "Optimizer learning rate actor : {}, critic: {}".format(self.lr_actor, self.lr_critic),
                              "=======================================================================================",
                              "Started training at (GMT) : {}".format(self.start_time),
                              "========================================================================================"
                              ]

        ############################## Training/Testing Results Saving/Loading ##############################

        self.NumOfTestBatches = 1  # batch
        self.TestCheckPoint = 10  # batches

        self.Train_Results = []  # [LocalizationTime, TotalDistance, Reward, SuccessFail] for each episode
        self.Test_Results = []  # [LocalizationTime, TotalDistance, Reward, SuccessFail] for each episode

        self.Episode = 0
        self.StartingBatch = 0

        # load existing model/results if exist
        if os.path.exists(self.checkpoint_path) and self.training_seed:
            outputt = "Loading previous results and existing model as a seed: {}".format(self.checkpoint_path)
            print(outputt)
            self.ConsoleOutput.append(outputt)
            self.PPO.load(self.checkpoint_path)
            Train_Results = np.load(self.directory + self.fileName + "_TrainingResults.npy")
            Test_Results = np.load(self.directory + self.fileName + "_TestingResults.npy")
            self.StartingBatch = Train_Results[:, 0].sum()//self.batch_size
            self.Episode = len(Train_Results)
            self.Train_Results = Train_Results.tolist()
            self.Test_Results = Test_Results.tolist()

        print('\n'.join(map(str, self.ConsoleOutput)))
        with open('ConsoleOutput.txt', 'a') as f:
            f.write('\n'.join(map(str, self.ConsoleOutput)))

    def sample(self) -> (Dict[str, np.ndarray], List):
        rewards = np.zeros(self.n_workers)
        infos = [{} for _ in range(self.n_workers)]

        Total_Reward = 0
        Total_Success = 0
        Total_Episodes = 0
        Total_Steps = 0
        Total_Cost = 0

        WorkersNewEpisode = np.ones(self.n_workers, dtype=bool)

        for t in range(self.worker_steps):

            actions = self.PPO.select_action(self.obs, self.Walls, self.InvalidActions, self.Learn, t, self.done)

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            for w, worker in enumerate(self.workers):
                self.obs[w], self.Walls[w], rewards[w], self.done[w], infos[w], self.InvalidActions[w] = \
                    worker.child.recv()

            for i in range(self.NumOfAgents):
                self.PPO.buffer.rewards[:, i, t] = rewards
                self.PPO.buffer.is_terminals[:, i, t] = self.done

            WorkersNewEpisode = self.done.astype(bool)
            if np.sum(self.done) > 0:
                Total_Episodes += np.sum(self.done)
                indices = np.argwhere(self.done)
                for i in indices:
                    current_ep_distance = infos[i[0]]['Movement Cost']
                    current_ep_reward = infos[i[0]]['reward']
                    current_ep_success = infos[i[0]]['Success']
                    current_ep_length = infos[i[0]]['length']
                    self.Train_Results.append([current_ep_length, current_ep_distance, current_ep_reward,
                                               current_ep_success])
                    Total_Reward += current_ep_reward
                    Total_Success += current_ep_success
                    Total_Steps += current_ep_length
                    Total_Cost += current_ep_distance

        return Total_Episodes, Total_Reward / Total_Episodes, Total_Success / Total_Episodes, Total_Steps / \
            Total_Episodes, Total_Cost / Total_Episodes

    def train(self):
        self.PPO.update()

    def run_training_loop(self):

        log_running_reward = 0
        log_running_batches = 0

        print_running_reward = 0
        print_running_success = 0
        print_running_batches = 0
        print_running_steps = 0
        print_running_cost = 0

        prev_training_reward = float('-inf')
        current_training_reward = 0
        num_training_batches = 0

        self.ConsoleOutput = []
        for update in range(self.StartingBatch, self.updates):
            # sample with current policy
            TotalEpisodes, AvgReward, AvgSuccess, AvgTime, AvgCost = self.sample()
            self.Episode += TotalEpisodes

            log_running_reward += AvgReward
            log_running_batches += 1

            print_running_reward += AvgReward
            print_running_success += AvgSuccess
            print_running_batches += 1
            print_running_steps += AvgTime
            print_running_cost += AvgCost

            current_training_reward += AvgReward
            num_training_batches += 1

            # train the model
            self.train()

            if (update+1) % self.print_freq == 0 and print_running_batches > 0:
                AverageBatchReward = print_running_reward/print_running_batches
                AverageBatchSuccess = print_running_success/print_running_batches
                AverageBatchSteps = print_running_steps/print_running_batches
                AverageBatchCost = print_running_cost / print_running_batches

                CurrStep = (update + 1) * self.batch_size
                outputt = "\nEpisode: %d, TimeStep: %d, AverageReward: %.2f, AverageSteps: %.2f, AverageCost: %.2f, " \
                          "AverageSuccess: %.2f, Elapsed time: %s" % (self.Episode, CurrStep,
                            AverageBatchReward, AverageBatchSteps, AverageBatchCost, AverageBatchSuccess,
                            datetime.now().replace(microsecond=0) - self.start_time)

                print_running_reward = 0
                print_running_success = 0
                print_running_batches = 0
                print_running_steps = 0
                print_running_cost = 0
                self.ConsoleOutput.append(outputt)
                print(outputt)

            if (update+1) % self.TestCheckPoint == 0:
                self.PPO.save(self.checkpoint_path)
                TestResult = self.test()
                outputt = ["################################### Testing ###################################",
                           "Average Testing Reward: {}, Average Testing Success: {}, Average Testing Steps: {}".format(
                               TestResult[0], TestResult[1], TestResult[2]),
                           "###############################################################################"]

                print('\n'.join(map(str, outputt)))
                self.ConsoleOutput += outputt

            if (update+1) % self.save_freq == 0:
                outputt = ["------------------------------------------------------------------------------------------",
                           "saving model and Results at : {}".format(self.checkpoint_path)]
                self.PPO.save(self.checkpoint_path)
                np.save(self.directory + self.fileName + "_TrainingResults.npy", self.Train_Results)
                np.save(self.directory + self.fileName + "_TestingResults.npy", self.Test_Results)
                outputt.append("model saved")
                outputt.append("Elapsed Time  : {}".format(datetime.now().replace(microsecond=0) - self.start_time))
                outputt.append("Training Rewards saved in: {}".format(self.directory + self.fileName))
                outputt.append("--------------------------------------------------------------------------------------")
                self.ConsoleOutput = self.ConsoleOutput + outputt
                print('\n'.join(map(str, outputt)))

            if (update+1) % self.log_freq == 0 and log_running_batches > 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_batches
                log_avg_reward = round(log_avg_reward, 4)

                self.log_f.write('{},{},{}\n'.format(self.Episode, (update+1)*self.batch_size, log_avg_reward))
                self.log_f.flush()

                log_running_reward = 0
                log_running_batches = 0

            if (update+1) % self.RewardCheck == 0:
                RewardCheck = current_training_reward / num_training_batches
                if RewardCheck > prev_training_reward or (RewardCheck > -80):
                    prev_training_reward = RewardCheck
                    current_training_reward = 0
                    num_training_batches = 0
                else:
                    outputtt = "################################################### RESET EVERYTHING AND START ALL " \
                               "OVER ############################\n"
                    print(outputtt)
                    self.ConsoleOutput.append(outputtt)

                    with open('ConsoleOutput.txt', 'a') as f:
                        f.write('\n'.join(map(str, self.ConsoleOutput)))

                    if os.path.exists(self.checkpoint_path):
                        os.remove(self.checkpoint_path)
                    return

            if (update+1) % self.save_console_freq == 0:
                with open('ConsoleOutput.txt', 'a') as f:
                    f.write('\n'.join(map(str, self.ConsoleOutput)))
                self.ConsoleOutput = []

    def test(self):
        testing_workers = [Worker(self.NumOfAgents, self.NumOfTargets, self.MapDimensions,
                                  self.NumOfActions, self.NumOfWalls, self.AreaSize, self.AgentWindow, self.Speed,
                                  self.DistanceThreshold, self.Idle, self.n_observations, self.max_ep_len, self.WallsDataset) for _ in
                           range(self.n_workers)]

        obs = np.zeros((self.n_workers, self.NumOfAgents, self.n_observations, self.state_dim[0],
                        self.state_dim[1]))
        Walls = np.zeros((self.n_workers, 1, self.MapDimensions[0], self.MapDimensions[1]))
        InvalidActions = np.zeros((self.n_workers, self.NumOfAgents, self.NumOfActions))
        done = np.ones(self.n_workers, dtype=int)

        Learn = 0

        for worker in testing_workers:
            worker.child.send(("reset", None))

        for i, worker in enumerate(testing_workers):
            obs[i], InvalidActions[i], Walls[i] = worker.child.recv()

        PPO_test = PPO(self.state_dim, self.MapDimensions, self.action_dim, self.lr_actor, self.lr_critic, self.gamma,
                       self.lmbda, self.epochs, self.eps_clip, self.n_workers, self.worker_steps, self.n_observations,
                       self.NumOfAgents, self.EncoderLocation, self.EncoderDims)

        PPO_test.load(self.checkpoint_path)

        rewards = np.zeros(self.n_workers)
        infos = [{} for _ in range(self.n_workers)]

        Total_Reward = 0
        Total_Episodes = 0
        Total_Success = 0
        Total_Steps = 0

        for t in range(self.worker_steps * self.NumOfTestBatches):

            actions = PPO_test.select_action(obs, Walls, InvalidActions, Learn, t, done)

            for w, worker in enumerate(testing_workers):
                worker.child.send(("step", actions[w]))

            for w, worker in enumerate(testing_workers):
                obs[w], Walls[w], rewards[w], done[w], infos[w], InvalidActions[w] = \
                    worker.child.recv()

            if np.sum(done) > 0:
                Total_Episodes += np.sum(done)
                indices = np.argwhere(done)
                for i in indices:
                    current_ep_distance = infos[i[0]]['Movement Cost']
                    current_ep_reward = infos[i[0]]['reward']
                    current_ep_success = infos[i[0]]['Success']
                    current_ep_length = infos[i[0]]['length']
                    self.Test_Results.append([self.Episode, current_ep_length, current_ep_distance, current_ep_reward,
                                              current_ep_success])
                    Total_Reward += current_ep_reward
                    Total_Success += current_ep_success
                    Total_Steps += current_ep_length

        for worker in testing_workers:
            worker.child.send(("close", None))

        return Total_Reward/Total_Episodes, Total_Success/Total_Episodes, Total_Steps/Total_Episodes

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))


if __name__ == "__main__":
    while 1:
        try:
            m = Main()
            m.run_training_loop()
            m.destroy()
        except:
            break
