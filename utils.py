import time
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import seaborn as sns
import math
import torch
from torch import nn
from PIL import Image


def ComputeAgentReadings(AgentLocation, AgentLocationGrid, TargetLocations, TargetLocationGrid, deltaT,
                         LearningStepSize, TargetsStrengths, NumOfTargets,
                         DetectorArea, AbsorptionRate, Walls, Scale, WallsPoints):
    oneSecTimeSteps = 1 / (deltaT / 1000)
    TotalTimeInSecs = LearningStepSize
    TotalTimeStepsPerSecond = TotalTimeInSecs * oneSecTimeSteps

    sourceEmissionRatePerSecond = TargetsStrengths / 60  # Source intensity
    sourceEmissionRatePer_delta_t = sourceEmissionRatePerSecond / TotalTimeStepsPerSecond

    AgentReadings = 0
    for i in range(NumOfTargets):
        DistanceToTarget = EuclideanDistance(AgentLocation, TargetLocations[0])
        NumOfWalls = CheckWalls(AgentLocation, TargetLocations[i], Scale, WallsPoints)

        if DistanceToTarget == 0:
            Lambda = sourceEmissionRatePer_delta_t
        else:
            Lambda = (DetectorArea * sourceEmissionRatePer_delta_t * (1 - NumOfWalls * AbsorptionRate)) \
                     / ((100 * DistanceToTarget) ** 2)

        p0 = np.exp(-Lambda * deltaT)
        p1 = 1 - p0
        # AgentReadings = 0
        # for j in range(int(TotalTimeStepsPerSecond)):
        #     if p0 <= random.random():
        #         AgentReadings += 1
        AgentReadings = p1 * TotalTimeStepsPerSecond

    return AgentReadings


def CheckWalls(LocationA, LocationB, Scale, WallsPoints):

    TempWallPoints = list(WallsPoints)
    WallsCount = 0
    angle = math.atan2(LocationB[1]-LocationA[1], LocationB[0]-LocationA[0])
    speed = 0.5 * Scale[0]
    x_disp = speed * math.cos(angle)
    y_disp = speed * math.sin(angle)
    actual_disp = math.sqrt((LocationB[1]-LocationA[1])**2 + (LocationB[0]-LocationA[0])**2)
    disp = speed
    TempLocation = LocationA
    Visited = []
    while disp < actual_disp:
        disp += speed
        TempLocation = (TempLocation[0]+x_disp, TempLocation[1]+y_disp)
        Gridx = math.floor(TempLocation[0] / Scale[0])
        Gridy = math.floor(TempLocation[1] / Scale[1])

        if (Gridx, Gridy) in Visited:
            continue

        Visited.append((Gridx, Gridy))
        for wall in TempWallPoints:
            if (Gridx, Gridy) in wall:
                WallsCount += 1
                TempWallPoints.remove(wall)
                break

    return WallsCount


def intersect(A, B, C, D):
    # Return true if line segments AB and CD intersect
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def manhattan_distance(pointA, pointB):
    ManDist = abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1])
    return ManDist


def EuclideanDistance(pointA, pointB):
    summ = 0
    for i in range(len(pointA)):
        summ += (pointA[i] - pointB[i]) ** 2

    Distance = np.sqrt(summ)
    return Distance


def PlotHeatmap(heatmap):  # just plots heatmap
    sns.heatmap(heatmap, cmap='viridis')


def PlotHeatmap2(heatmap):  # plots heatmap with numbers
    sns.heatmap(heatmap, cmap='viridis', annot=True, fmt=".3g", linewidths=.5)  # .5f -> 5 dec, 3g -> 3 sig. figs


def MapCropping(Map, MapDims, Window, AgentPrevLocation):
    WLr = (Window[0] - 1) / 2
    WLc = (Window[1] - 1) / 2
    row = AgentPrevLocation[0]
    col = AgentPrevLocation[1]

    diffU = (min(WLr + row, MapDims[0] - 1) - row) - WLr
    diffD = (row - max(row - WLr, 0)) - WLr
    diffR = (min(WLc + col, MapDims[1] - 1) - col) - WLc
    diffL = (col - max(col - WLc, 0)) - WLc

    U = int(min(WLr + row, MapDims[0] - 1) - diffD) + 1
    D = int(max(row - WLr, 0) + diffU)
    R = int(min(WLc + col, MapDims[1] - 1) - diffL) + 1
    L = int(max(col - WLc, 0) + diffR)

    NewMap = Map[D:U, L:R]

    return NewMap


def FindDistanceToClosest(Target, Locations):
    distance = float('inf')
    for i in range(len(Locations)):
        CurrentDistance = manhattan_distance(Target, Locations[i])
        if CurrentDistance < distance:
            distance = CurrentDistance

    return distance


def FindDistanceToClosest2(Target, Locations):
    distance = float('inf')
    for i in range(len(Locations)):
        CurrentDistance = EuclideanDistance(Target, Locations[i])
        if CurrentDistance < distance:
            distance = CurrentDistance

    return distance


def SmoothMap(Map):
    SmoothedMap = gaussian_filter(Map, sigma=1)
    return SmoothedMap


def ClosestPointOnLine(p1, p2, p3):
    if p1 == p2:
        return p1

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
    p4 = x1 + a * dx, y1 + a * dy
    InBetween = int(EuclideanDistance(p4, p1) + EuclideanDistance(p4, p2)) == int(EuclideanDistance(p1, p2))
    if InBetween:
        return p4
    else:
        if EuclideanDistance(p3, p1) < EuclideanDistance(p3, p2):
            return p1
        else:
            return p2


def Resize(Map, size):
    newMap = Map * 255
    newMap = Image.fromarray(newMap)
    newMap = newMap.resize(size, Image.ANTIALIAS)
    newMap = np.array(newMap) / 255

    return newMap


def Resize2(Map, size):
    newMap = resize(Map, size, anti_aliasing=True, anti_aliasing_sigma=2)
    return newMap


def AllWalls(Walls, MapDims):
    All_Walls = np.zeros(MapDims)
    All_Walls_Points = [[] for i in range(len(Walls))]

    for i in range(MapDims[0]):
        for j in range(MapDims[1]):
            for k in range(len(Walls)):
                ItIsAWall = False
                if Walls[k][0][0] == Walls[k][1][0] and i == Walls[k][0][0]:
                    ItIsAWall = max(Walls[k][0][1], Walls[k][1][1]) >= j >= min(Walls[k][0][1],
                                                                                Walls[k][1][1])

                elif Walls[k][0][1] == Walls[k][1][1] and j == Walls[k][0][1]:
                    ItIsAWall = max(Walls[k][0][0], Walls[k][1][0]) >= i >= min(Walls[k][0][0],
                                                                                Walls[k][1][0])

                if ItIsAWall:
                    All_Walls[i][j] = 1
                    All_Walls_Points[k].append((i, j))
                    break

    return All_Walls, All_Walls_Points