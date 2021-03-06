#! python

import csv
import glob
import random
import math
import operator
from functools import reduce
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from os import system, name 
import numpy as np


def clear():
    '''
    define console clear function
    '''
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

def GetInputDataFile():
    '''
    get user input for which data file to run algo on
    also get number of centroids to compute and whether to
    save scatter plot images or not
    '''
    #clear()
    dataFile = None
    k = None
    csvList = glob.glob("data/*.csv")
    print("select a data file to run spectral clustering")
    for idx, filePath in enumerate(csvList):
        print(f'({idx}) {filePath}')
    dataFileIndex = int(input("select option "))
    if 0 <= dataFileIndex < len(csvList):
        dataFile = csvList[dataFileIndex]
    else:
        GetInputDataFile()

    sigma = float(input("enter sigma value for gaussian similarity function "))
    k = int(input("enter number of clusters to compute "))
    YES_VALUES = {'y', 'yes', 'Y'}
    saveScatterPlots = input("save scatter plot for each iteration ? (y,N) ").lower() in YES_VALUES
    if(saveScatterPlots):
        print('scatter plots will be saved in ./images/ folder')

    print('output csv files will be store in ./output/ folder')
    return (dataFile, k, saveScatterPlots, sigma)

def GetDistance(x, y):
    '''
    calculate Euclidean distance between two n dimentional points
    '''
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

def Assign(centroides, data):
    '''
    Assign each point to one of the k clusters
    '''
    mapping = []
    for point in data:
        computedMap = {
            'closestDistance' : None,
            'closestCentroid' : None,
            'point' : point,
        }
        for centroid in centroides:
            distance = GetDistance(point, centroid)
            if computedMap['closestDistance'] == None or computedMap['closestDistance'] > distance :
                computedMap['closestDistance'] = distance
                computedMap['closestCentroid'] = centroid

        mapping.append(computedMap)
    return mapping

def sumPoints(x1,x2):
    return [(x1[0]+x2[0]),(x1[1]+x2[1])]

def Update(centroides, previousIterationData):
    '''
    calculate new centroid based on clusters
    '''
    ## compute mean of each cluster, make that the new starting points
    differenceVector = []
    newCentroides = []
    for centroid in centroides:
        cdata = [y['point'] for y in list(filter(lambda x: x['closestCentroid'] == centroid, previousIterationData))]
        totalVector = reduce(sumPoints, cdata)
        mean = [x / len(cdata) for x in totalVector]
        print(f'number of data points for {centroid} are {len(cdata)} with mean {mean}')

        newCentroides.append(mean)
        distance = GetDistance(mean, centroid)
        differenceVector.append(distance)
    return (newCentroides, differenceVector)

def CalculatePartitions(data, k, epsilon, maxIterations):
    '''
    cluster data in k centroids based on lloyds algorithm
    '''
    ## ramdomly select K starting points to start
    centroides = random.sample(data, k)
    print(f'assigning {len(data)} number of data points to {k} clusters')
    mapped = Assign(centroides, data)
    significientDifference = True
    itr=1
    ## repeat unit no change in clusters
    while significientDifference:
        print('iteration', itr)
        itr += 1
        newCentroides, diffVector = Update(centroides, mapped)
        if sum(diffVector) < epsilon or itr > maxIterations:
            significientDifference = False
        if significientDifference:
            centroides = newCentroides
            mapped = Assign(centroides, data)
        
    C = [centroides.index(x['closestCentroid'])+1 for x in mapped]
    # OUTPUT
    # (i) - centroides matrix 
    # (ii) - cluster index vector C ∈{ 1,2,3…K }^N, Where C(i)=j indicates that the ith row of X belongs to cluster j
    return (centroides, C)

colors = list(mcolors.CSS4_COLORS.keys())


def plotCluster(originalData, C):
    title = 'clusters computed with spectral clustring'
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.set(title=title)
    for idx, val in enumerate(originalData):
        ax.scatter(val[0], val[1], color=colors[C[idx]+10], s=50)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f'images/{title}.png')
    plt.show()
    plt.close()

def GaussianSimilarityFunction(p1, p2, sigma):
    disSquare = sum([(a - b) ** 2 for a, b in zip(p1, p2)])/(2*pow(sigma, 2))
    similaritySocre = math.exp(-1*disSquare)
    return similaritySocre

def DiagMatrixHelper(x, y, AdjacencyMatrix):
    if x!=y :
        return 0
    else :
        return sum(AdjacencyMatrix[x])


def GetEigneVectorsForSmallEigenValues(a, k):
    eigenValues, eigenVectors = np.linalg.eig(a)
    idx = eigenValues.argsort()[::1] # sort smallest to largest eignevalue
    idx = idx[0:k] # take indexes corrosponding to k smallest eignevalue 
    eigenValues = eigenValues[idx]
    print(eigenValues)  
    eigenVectors = eigenVectors[idx] # order corresponding eignevectors
    return eigenVectors

def ComputeClustering(data, k=2, sigma=0.3, epsilon=10**-5, maxIterations=50):
    dataCount = len(data)
    # Compute adjacency matrix based on Gaussian Similarity function
    AdjacencyMatrix = [[GaussianSimilarityFunction(data[x], data[y], sigma) for x in range(dataCount)] for y in range(dataCount)] 
    # Compute diagonal matrix from adjacency matrix
    DiagonalMatrix = [[DiagMatrixHelper(x,y, AdjacencyMatrix) for x in range(dataCount)] for y in range(dataCount)]
    # Compute laplacian matrix L = D - A
    LaplacianMatrix = [[ DiagonalMatrix[x][y]-AdjacencyMatrix[x][y] for x in range(dataCount)] for y in range(dataCount)] # DiagonalMatrix - AdjacencyMatrix
    # Compute eigen vectors corrosponding to k smallest eigen values
    eigen = GetEigneVectorsForSmallEigenValues(LaplacianMatrix, k)
    # Arrange egien vectors as column in a new U matrix
    U = np.transpose(eigen).tolist()
    # Perform K mean clustering on U
    _, C = CalculatePartitions(U, k, epsilon, maxIterations)
    return (C, AdjacencyMatrix, U)
    
if __name__ == "__main__":
    print("Spectral clustering")
    dataFile, k, savePlot, sigma = GetInputDataFile()
    print(f"reading file from {dataFile}")
    
    data = []
    with open(dataFile, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            dataRow = [float(row[0]),float(row[1])]
            data.append(dataRow)
    
    C,_,_ = ComputeClustering(data, k, sigma)

    if savePlot:
        plotCluster(data, C)
