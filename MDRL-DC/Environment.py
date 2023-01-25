import math
import time

import numpy
import random
import numpy as np
import utils
from matplotlib import pyplot as plt
import torch


class Environment:

    def __init__(self, NumOfAgents=1, NumOfTargets=1, MapDimensions=(10, 10), NumOfActions=5, NumOfWalls=2,
                 AreaSize=(10, 10), AgentWindow=(5, 5), Speed=5, DistanceThreshold=50, Idle=1, n_observations=4,
                 EpisodeLength=100, WallsDataset=()):
        self.NumOfAgents = NumOfAgents
        self.NumOfTargets = NumOfTargets
        self.MapDimensions = MapDimensions
        self.AgentWindow = AgentWindow
        self.done = False
        self.Found = [0] * self.NumOfTargets
        self.AreaSize = AreaSize
        self.NumOfActions = NumOfActions
        self.CombinedActions = self.NumOfActions ** self.NumOfAgents
        self.n_observations = n_observations
        self.Scale = [self.AreaSize[0] / self.MapDimensions[0], self.AreaSize[1] / self.MapDimensions[1]]
        self.DistanceThreshold = DistanceThreshold
        self.Idle = Idle
        self.MinInitialDistanceToTarget = 10
        self.EpisodeLength = EpisodeLength

        self.Speed = Speed  # m/s
        self.AgentsSpeeds = [Speed] * self.NumOfAgents  # m/s

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')

        self.deltaT = 10  # ms
        self.LearningStepSize = 1  # in seconds, this is the time of a step in an episode.
        self.TargetsStrengths = int(1e9)  # per minute, assuming all targets have the same strength
        self.AbsorptionRate = 0.1  # to be multiplied by the number of walls
        self.DetectorArea = 200  # cm^2
        self.MaxPossibleReading = (1000 / self.deltaT) * self.NumOfTargets

        self.TargetLocations = []
        self.TargetLocationsGrid = []
        self.AgentsLocations = []
        self.AgentsLocationsGrid = []
        self.AgentsPrevLocations = []
        self.AgentsPrevLocationsGrid = []


        self.CurrentReadings = [-1] * self.NumOfAgents

        self.observation_space_size = (AgentWindow[0], AgentWindow[1])

        self.LocationsMap = np.zeros((self.NumOfAgents, self.MapDimensions[0], self.MapDimensions[1]))
        self.VisitCounts = np.zeros((1, self.MapDimensions[0], self.MapDimensions[1]))
        self.ReadingsMap = np.zeros((1, self.MapDimensions[0], self.MapDimensions[1]))
        self.OtherLocations = np.zeros((self.NumOfAgents, self.MapDimensions[0], self.MapDimensions[1]))
        self.RewardsMap = np.zeros((self.MapDimensions[0], self.MapDimensions[1]))

        self.NumOfWalls = NumOfWalls
        self.WallsDataset = WallsDataset

        self.Walls = np.zeros(self.MapDimensions)
        self.WallsPoints = []

        if self.NumOfWalls > 0:
            self.Walls, self.WallsPoints = self.MakeWalls()
        self.Walls = np.expand_dims(self.Walls, axis=0)

        self.HistoryPath = []
        self.HistoryPathGrid = []

        self.InitialState()

    def InitialState(self):
        self.done = False
        self.HistoryPath = []
        self.HistoryPathGrid = []

        for i in range(self.NumOfTargets):
            Tx = random.randint(0, self.AreaSize[0] - 1)
            Ty = random.randint(0, self.AreaSize[1] - 1)
            Tx_G = math.floor(Tx / self.Scale[0])
            Ty_G = math.floor(Ty / self.Scale[1])
            TargetLocationGrid = (Tx_G, Ty_G)
            while (TargetLocationGrid in self.TargetLocationsGrid) or (self.Walls[0][Tx_G][Ty_G] == 1):
                Tx = random.randint(0, self.AreaSize[0] - 1)
                Ty = random.randint(0, self.AreaSize[1] - 1)
                Tx_G = math.floor(Tx / self.Scale[0])
                Ty_G = math.floor(Ty / self.Scale[1])
                TargetLocationGrid = (Tx_G, Ty_G)
            self.TargetLocations.append((Tx, Ty))
            self.TargetLocationsGrid.append(TargetLocationGrid)

        if self.NumOfWalls > 0:
            self.RewardsMap = self.ComputeRewardsMap3(self.TargetLocationsGrid[0], self.TargetLocations[0])
        else:
            self.RewardsMap = self.ComputeRewardsMap2(self.TargetLocationsGrid[0])

        for i in range(self.NumOfAgents):
            xL = random.randint(0, self.AreaSize[0] - 1)
            yL = random.randint(0, self.AreaSize[1] - 1)
            xL_G = math.floor(xL / self.Scale[0])
            yL_G = math.floor(yL / self.Scale[1])
            AgentLocationGrid = (xL_G, yL_G)
            while (AgentLocationGrid in self.TargetLocationsGrid) or (AgentLocationGrid in self.AgentsLocationsGrid) \
                    or (self.Walls[0][xL_G][yL_G] == 1) or (self.RewardsMap[xL_G, yL_G] <
                                                            self.MinInitialDistanceToTarget):
                xL = random.randint(0, self.AreaSize[0] - 1)
                yL = random.randint(0, self.AreaSize[1] - 1)
                xL_G = math.floor(xL / self.Scale[0])
                yL_G = math.floor(yL / self.Scale[1])
                AgentLocationGrid = (xL_G, yL_G)
            self.AgentsLocations.append((xL, yL))
            self.AgentsLocationsGrid.append(AgentLocationGrid)

        self.AgentsPrevLocations = self.AgentsLocations
        self.AgentsPrevLocationsGrid = self.AgentsLocationsGrid

        self.HistoryPath.append(self.AgentsLocations)
        self.HistoryPathGrid.append(self.AgentsLocationsGrid)

    def reset(self):
        self.AgentsLocations = []
        self.TargetLocations = []
        self.TargetLocationsGrid = []
        self.AgentsLocationsGrid = []
        self.AgentsPrevLocations = []
        self.AgentsPrevLocationsGrid = []
        self.done = False
        self.Found = [0] * self.NumOfTargets
        self.MaxPossibleReading = (1000 / self.deltaT) * self.NumOfTargets

        self.WallLengths = [random.randint(round(self.MapDimensions[0] / 2), self.MapDimensions[0] - 3) for i in
                            range(self.NumOfWalls)]

        self.Walls = np.zeros(self.MapDimensions)
        self.WallsPoints = []

        if self.NumOfWalls > 0:
            self.Walls, self.WallsPoints = self.MakeWalls()
        self.Walls = np.expand_dims(self.Walls, axis=0)

        self.LocationsMap = np.zeros((self.NumOfAgents, self.MapDimensions[0], self.MapDimensions[1]))
        self.VisitCounts = np.zeros((1, self.MapDimensions[0], self.MapDimensions[1]))
        self.ReadingsMap = np.zeros((1, self.MapDimensions[0], self.MapDimensions[1]))
        self.OtherLocations = np.zeros((self.NumOfAgents, self.MapDimensions[0], self.MapDimensions[1]))
        self.RewardsMap = np.zeros((self.MapDimensions[0], self.MapDimensions[1]))

        self.InitialState()

        for i in range(self.NumOfAgents):
            self.CurrentReadings[i] = utils.ComputeAgentReadings(self.AgentsLocations[i], self.AgentsLocationsGrid[i],
                                                                 self.TargetLocations, self.TargetLocationsGrid,
                                                                 self.deltaT, self.LearningStepSize,
                                                                 self.TargetsStrengths, self.NumOfTargets,
                                                                 self.DetectorArea, self.AbsorptionRate,
                                                                 self.Walls, self.Scale,
                                                                 self.WallsPoints)

            # if self.CurrentReadings[i] > 0.75 * self.MaxPossibleReading:
            #     if self.AgentsSpeeds[i] == self.Speed:
            #         self.AgentsSpeeds[i] = round(0.3 * self.Speed)
            # else:
            #     self.AgentsSpeeds[i] = self.Speed

            self.ReadingsMap[0][self.AgentsLocationsGrid[i][0]][self.AgentsLocationsGrid[i][1]] = self.CurrentReadings[
                i]
            self.VisitCounts[0][self.AgentsLocationsGrid[i][0]][self.AgentsLocationsGrid[i][1]] += 1

        All_obs = np.zeros((self.NumOfAgents, self.n_observations, self.observation_space_size[0],
                            self.observation_space_size[1]))
        All_InvalidActions = np.zeros((self.NumOfAgents, self.NumOfActions), dtype=numpy.int)
        for i in range(self.NumOfAgents):
            All_obs[i, :] = self.getObservation(i)
            _, InvalidActions = self.CheckPossibleActions(self.AgentsLocations[i], self.AgentsLocationsGrid[i], i)
            All_InvalidActions[i] = InvalidActions

        return All_obs, All_InvalidActions, self.Walls

    def getObservation(self, i):

        self.LocationsMap[i][self.AgentsLocationsGrid[i][0]][self.AgentsLocationsGrid[i][1]] = 1

        for j in range(self.NumOfAgents):
            if j == i:
                continue
            self.OtherLocations[i][self.AgentsLocationsGrid[j][0]][self.AgentsLocationsGrid[j][1]] += 1

        AgentPrevLocationCroppingIndex = -1
        AgentPrevLocation = self.HistoryPathGrid[AgentPrevLocationCroppingIndex][i]

        # AgentWindowedLocation
        WindowedLocation = \
            utils.MapCropping(self.LocationsMap[i], self.MapDimensions, self.AgentWindow, AgentPrevLocation)

        # AgentWindowedVisitCounts
        WindowedVisitCounts = utils.MapCropping(self.VisitCounts[0], self.MapDimensions, self.AgentWindow,
                                                AgentPrevLocation)
        if np.max(WindowedVisitCounts) > 0:
            WindowedVisitCounts = WindowedVisitCounts / np.max(self.VisitCounts[0])

        # AgentWindowedReadings
        WindowedReadings = utils.MapCropping(self.ReadingsMap[0], self.MapDimensions, self.AgentWindow,
                                             AgentPrevLocation)
        WindowedReadings = WindowedReadings / self.MaxPossibleReading

        # WindowedWalls
        WindowedWalls = utils.MapCropping(self.Walls[0], self.MapDimensions, self.AgentWindow,
                                          AgentPrevLocation)
        # AgentLocation
        LocationsMap = self.LocationsMap[i]
        LocationsMap = utils.Resize(LocationsMap, self.AgentWindow)

        # AllVisitCounts
        VisitCounts = self.VisitCounts[0] / np.max(self.VisitCounts[0])
        VisitCounts = utils.Resize(VisitCounts, self.AgentWindow)

        # AllReadingsMap
        ReadingsMap = self.ReadingsMap[0] / self.MaxPossibleReading
        ReadingsMap = utils.Resize(ReadingsMap, self.AgentWindow)

        # AllOtherLocations
        OtherLocations = self.OtherLocations[i] / np.max(self.OtherLocations[i])
        OtherLocations = utils.Resize(OtherLocations, self.AgentWindow)

        # plt.figure()
        # plt.subplot(2, 3, 1)
        # utils.PlotHeatmap2(self.LocationsMap[i])
        # plt.subplot(2, 3, 2)
        # utils.PlotHeatmap2(self.OtherLocations[i])
        # plt.subplot(2, 3, 3)
        # utils.PlotHeatmap2(self.VisitCounts[0])
        # plt.subplot(2, 3, 4)
        # utils.PlotHeatmap2(self.ReadingsMap[0])

        #
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # utils.PlotHeatmap2(LocationsMap)
        # plt.subplot(2, 2, 2)
        # utils.PlotHeatmap2(OtherLocations)
        # plt.subplot(2, 2, 3)
        # utils.PlotHeatmap2(VisitCounts)
        # plt.subplot(2, 2, 4)
        # utils.PlotHeatmap2(ReadingsMap)
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # utils.PlotHeatmap2(WindowedLocation)
        # plt.subplot(2, 2, 2)
        # utils.PlotHeatmap2(WindowedVisitCounts)
        # plt.subplot(2, 2, 3)
        # utils.PlotHeatmap2(WindowedReadings)
        # plt.show()

        WindowedLocation = np.expand_dims(WindowedLocation, 0)
        WindowedVisitCounts = np.expand_dims(WindowedVisitCounts, 0)
        WindowedReadings = np.expand_dims(WindowedReadings, 0)
        WindowedWalls = np.expand_dims(WindowedWalls, 0)
        LocationsMap = np.expand_dims(LocationsMap, 0)
        VisitCounts = np.expand_dims(VisitCounts, 0)
        ReadingsMap = np.expand_dims(ReadingsMap, 0)
        OtherLocations = np.expand_dims(OtherLocations, 0)

        obs = np.append(WindowedLocation, WindowedVisitCounts, axis=0)
        obs = np.append(obs, WindowedReadings, axis=0)
        obs = np.append(obs, WindowedWalls, axis=0)
        obs = np.append(obs, LocationsMap, axis=0)
        obs = np.append(obs, VisitCounts, axis=0)
        obs = np.append(obs, ReadingsMap, axis=0)
        obs = np.append(obs, OtherLocations, axis=0)
        obs = np.expand_dims(obs, 0)

        return obs

    def step(self, actions):

        # print(actions)
        reward = 0
        AgentsNewLocations = []
        AgentsNewLocationsGrid = []

        for i in range(self.NumOfAgents):

            AgentNewLocation, AgentNewLocationGrid = self.AgentNextLocation(self.AgentsLocations[i], actions[i], i)

            AgentNewLocation = (max(0, min(AgentNewLocation[0], self.AreaSize[0]-1)), max(0, min(AgentNewLocation[1],
                                                                                                 self.AreaSize[1]-1)))
            AgentNewLocationGrid = (max(0, min(AgentNewLocationGrid[0], self.MapDimensions[0]-1)),
                                    max(0, min(AgentNewLocationGrid[1], self.MapDimensions[1]-1)))

            if (self.Idle == 1) and (actions[i] == self.NumOfActions - 1):  # cost of taking action
                reward += 0
            else:
                reward += -1

            ClosestPointOnPath = utils.ClosestPointOnLine(self.AgentsLocations[i], AgentNewLocation,
                                                          self.TargetLocations[0])
            if utils.EuclideanDistance(ClosestPointOnPath, self.TargetLocations[0]) < self.DistanceThreshold and \
                    utils.CheckWalls(self.AgentsLocations[i], self.TargetLocations[0], self.Scale,
                                     self.WallsPoints) < 1:
                if self.Found[0] == 0:
                    self.Found[0] = 1
                    AgentNewLocation = ClosestPointOnPath
                    AgentNewLocationGrid = self.TargetLocationsGrid[0]
                    AgentsNewLocations.append(AgentNewLocation)
                    AgentsNewLocationsGrid.append(AgentNewLocationGrid)
                    continue

            AgentsNewLocations.append(AgentNewLocation)
            AgentsNewLocationsGrid.append(AgentNewLocationGrid)

        self.LocationsMap = np.zeros((self.NumOfAgents, self.MapDimensions[0], self.MapDimensions[1]))
        self.OtherLocations = np.zeros((self.NumOfAgents, self.MapDimensions[0], self.MapDimensions[1]))

        self.AgentsPrevLocations = self.AgentsLocations
        self.AgentsPrevLocationsGrid = self.AgentsLocationsGrid
        self.AgentsLocations = AgentsNewLocations
        self.AgentsLocationsGrid = AgentsNewLocationsGrid
        self.HistoryPath.append(self.AgentsLocations)
        self.HistoryPathGrid.append(self.AgentsLocationsGrid)

        if self.Found[0] == 1:
            reward += 50

        for i in range(self.NumOfAgents):
            self.CurrentReadings[i] = utils.ComputeAgentReadings(self.AgentsLocations[i], self.AgentsLocationsGrid[i],
                                                                 self.TargetLocations, self.TargetLocationsGrid,
                                                                 self.deltaT, self.LearningStepSize,
                                                                 self.TargetsStrengths, self.NumOfTargets,
                                                                 self.DetectorArea, self.AbsorptionRate,
                                                                 self.Walls, self.Scale,
                                                                 self.WallsPoints)

            # if self.CurrentReadings[i] > 0.75 * self.MaxPossibleReading:
            #     if self.AgentsSpeeds[i] == self.Speed:
            #         self.AgentsSpeeds[i] = round(0.3 * self.Speed)
            # else:
            #     self.AgentsSpeeds[i] = self.Speed

            self.ReadingsMap[0][self.AgentsLocationsGrid[i][0]][self.AgentsLocationsGrid[i][1]] = self.CurrentReadings[
                i]

            self.VisitCounts[0][self.AgentsLocationsGrid[i][0]][self.AgentsLocationsGrid[i][1]] += 1

            self.UpdateInBetween(actions[i], i)

        if np.prod(self.Found):
            self.done = True

        All_obs = np.zeros((self.NumOfAgents, self.n_observations, self.observation_space_size[0],
                            self.observation_space_size[1]))
        All_InvalidActions = np.zeros((self.NumOfAgents, self.NumOfActions), dtype=numpy.int)
        for i in range(self.NumOfAgents):
            All_obs[i, :] = self.getObservation(i)
            _, InvalidActions = self.CheckPossibleActions(self.AgentsLocations[i], self.AgentsLocationsGrid[i], i)
            All_InvalidActions[i] = InvalidActions

        HistoryLength = len(self.HistoryPath)
        if (self.HistoryPath[HistoryLength-1] == self.HistoryPath[HistoryLength-2]) and self.Idle:
            All_InvalidActions[:, -1] = 1

        return (All_obs, reward, self.done, All_InvalidActions)

    def AgentNextLocation(self, AgentsLocation, AgentAction, agent):
        if (self.Idle == 1) and (AgentAction == self.NumOfActions - 1):
            AgentNewLocation = AgentsLocation
            AgentNewLocationGrid = (
                math.floor(AgentNewLocation[0] / self.Scale[0]), math.floor(AgentNewLocation[1] / self.Scale[1]))
        else:
            angle = (AgentAction / (self.NumOfActions - self.Idle)) * 2 * math.pi
            x_disp = self.AgentsSpeeds[agent] * math.cos(angle)
            y_disp = self.AgentsSpeeds[agent] * math.sin(angle)
            x_loc = int(AgentsLocation[0] + x_disp)
            y_loc = int(AgentsLocation[1] + y_disp)

            AgentNewLocation = (x_loc, y_loc)
            AgentNewLocationGrid = (
                math.floor(AgentNewLocation[0] / self.Scale[0]), math.floor(AgentNewLocation[1] / self.Scale[1]))

        return AgentNewLocation, AgentNewLocationGrid

    def breadthFirstSearch(self, CurrentLocation, Target):
        """Search the shallowest nodes in the search tree first."""
        "*** YOUR CODE HERE ***"
        explored = []
        queue = [[CurrentLocation]]
        if CurrentLocation == Target:
            return 0

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                PossibleLocations = self.CheckPossibleNextLocations(node)
                for newLocation in PossibleLocations:
                    new_path = list(path)
                    new_path.append(newLocation)
                    queue.append(new_path)
                    if newLocation == Target:
                        return len(new_path) - 1
                explored.append(node)

        raise Exception("Reward: was not able to find shortest path to the target")

    def CheckPossibleNextLocations2(self, AgentLocation):
        gridX = [0, 0, -1, 1]
        gridY = [1, -1, 0, 0]
        PossibleNextLocations = []
        for i in range(4):
            nextX = int(min(self.MapDimensions[0] - 1, max(0, AgentLocation[0] + gridX[i])))
            nextY = int(min(self.MapDimensions[1] - 1, max(0, AgentLocation[1] + gridY[i])))
            if (nextX, nextY) not in PossibleNextLocations and self.Walls[0][nextX][nextY] != 1:
                PossibleNextLocations.append((nextX, nextY))

        return PossibleNextLocations

    def CheckPossibleNextLocations(self, AgentLocation):
        gridX = np.arange(-1, 2, 1)
        gridY = np.arange(-1, 2, 1)
        PossibleNextLocations = []
        for i in gridX:
            for j in gridY:
                if i == 0 and j == 0:
                    continue
                nextX = int(min(self.MapDimensions[0] - 1, max(0, AgentLocation[0] + i)))
                nextY = int(min(self.MapDimensions[1] - 1, max(0, AgentLocation[1] + j)))
                if (nextX, nextY) not in PossibleNextLocations and self.Walls[0][nextX][nextY] != 1:
                    PossibleNextLocations.append((nextX, nextY))

        return PossibleNextLocations

    def CheckPossibleActions(self, AgentLocation, AgentLocationGrid, agent):
        PossibleActions = np.zeros(self.NumOfActions, dtype=numpy.int)
        InvalidActions = np.zeros(self.NumOfActions, dtype=numpy.int)

        for i in range(self.NumOfActions):
            AgentNextLocation, AgentNextLocationGrid = self.AgentNextLocation(AgentLocation, i, agent)
            if (AgentNextLocation[0] < 0) or (AgentNextLocation[0] > self.AreaSize[0] - 1) or \
                    (AgentNextLocation[1] < 0) or (AgentNextLocation[1] > self.AreaSize[1] - 1):
                InvalidActions[i] = 1
            elif self.Walls[0][AgentNextLocationGrid[0]][AgentNextLocationGrid[1]] == 1:
                InvalidActions[i] = 1
            elif utils.CheckWalls(AgentNextLocation, AgentLocation, self.Scale, self.WallsPoints) > 0:
                InvalidActions[i] = 1
            else:
                PossibleActions[i] = 1

        return PossibleActions, InvalidActions

    def DecaySpeed(self, SpeedDecay, MinSpeed):
        self.Speed = max(MinSpeed, self.Speed - SpeedDecay)


    def ComputeRewardsMap2(self, TargetLocation):
        RewardsMap = np.zeros((self.MapDimensions[0], self.MapDimensions[1]))
        for i in range(RewardsMap.shape[0]):
            for j in range(RewardsMap.shape[1]):
                RewardsMap[i][j] = utils.EuclideanDistance((i, j), TargetLocation)
                #RewardsMap[i][j] = utils.manhattan_distance((i, j), TargetLocation)

        return RewardsMap

    def ComputeRewardsMap(self, TargetLocation):
        """Search the shallowest nodes in the search tree first."""
        "*** YOUR CODE HERE ***"
        RewardsMap = np.zeros((self.MapDimensions[0], self.MapDimensions[1]))
        explored = []
        queue = [[TargetLocation]]

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                RewardsMap[node[0]][node[1]] = len(path) - 1
                PossibleLocations = self.CheckPossibleNextLocations(node)
                for newLocation in PossibleLocations:
                    new_path = list(path)
                    new_path.append(newLocation)
                    queue.append(new_path)
                explored.append(node)

        return RewardsMap

    def ComputeRewardsMap3(self, TargetLocationGrid, TargetLocation):
        RewardsMap = np.zeros((self.MapDimensions[0], self.MapDimensions[1]))
        for i in range(RewardsMap.shape[0]):
            for j in range(RewardsMap.shape[1]):
                location = [(i+0.5)*self.Scale[0], (j+0.5)*self.Scale[1]]
                if utils.CheckWalls(location, TargetLocation, self.Scale, self.WallsPoints) > 0:
                    RewardsMap[i][j] = 10*self.MinInitialDistanceToTarget
                else:
                    RewardsMap[i][j] = utils.EuclideanDistance((i, j), TargetLocationGrid)
                #RewardsMap[i][j] = utils.manhattan_distance((i, j), TargetLocation)

        return RewardsMap

    def UpdateInBetween(self, action, i):
        if utils.manhattan_distance(self.AgentsLocationsGrid[i], self.AgentsPrevLocationsGrid[i]) > 1 and self.Speed > \
                self.Scale[0]:
            temp_speed = 0.5 * self.Speed / math.floor(self.Speed / self.Scale[0])
            angle = (action / (self.NumOfActions - self.Idle)) * 2 * math.pi
            x_disp = temp_speed * math.cos(angle)
            y_disp = temp_speed * math.sin(angle)
            temp_location = self.AgentsPrevLocations[i]
            VisitedGrids = [self.AgentsLocationsGrid[i], self.AgentsPrevLocationsGrid[i]]

            for j in range(round(self.Speed / temp_speed)):
                temp_location = (int(temp_location[0] + x_disp), int(temp_location[1] + y_disp))

                TempLocationGrid = (math.floor(temp_location[0] / self.Scale[0]), math.floor(temp_location[1] /
                                                                                             self.Scale[1]))

                if TempLocationGrid not in VisitedGrids:
                    if TempLocationGrid[0] >= self.MapDimensions[0] or TempLocationGrid[1] >= self.MapDimensions[1]:
                        continue
                    TempCurrentReading = utils.ComputeAgentReadings(temp_location, TempLocationGrid,
                                                                    self.TargetLocations, self.TargetLocationsGrid,
                                                                    self.deltaT, self.LearningStepSize,
                                                                    self.TargetsStrengths, self.NumOfTargets,
                                                                    self.DetectorArea, self.AbsorptionRate,
                                                                    self.Walls, self.Scale,
                                                                    self.WallsPoints)

                    self.ReadingsMap[0][TempLocationGrid[0]][TempLocationGrid[1]] = TempCurrentReading
                    self.VisitCounts[0][TempLocationGrid[0]][TempLocationGrid[1]] += 1
                    VisitedGrids.append(TempLocationGrid)

    def MakeWalls(self):
        WallsIndex = np.random.randint(0, len(self.WallsDataset[0]))
        Walls = self.WallsDataset[0][WallsIndex]
        WallsPoints = self.WallsDataset[1][WallsIndex]

        return Walls, WallsPoints

    def DistanceBetweenWalls(self, margin):
        for i in range(len(self.WallsPoints)):
            for j in range(i + 1, len(self.WallsPoints)):
                for k in range(len(self.WallsPoints[i])):
                    for l in range(len(self.WallsPoints[j])):
                        if utils.manhattan_distance(self.WallsPoints[i][k], self.WallsPoints[j][l]) < margin:
                            return False
        return True