import csv
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# This function takes as input the path to a csv file, and returns the datapoints as a list of dicts.
def load_data(filepath):
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        pkmnList = []
        for row in reader:
            pkmn = {'HP': str(row['HP']), 'Attack': str(row['Attack']), 'Defense': str(row['Defense']), 'Sp. Atk': str(row['Sp. Atk']), 'Sp. Def': str(row['Sp. Def']), 'Speed': str(row['Speed'])}
            pkmnList.append(pkmn)
        return pkmnList

# This function takes as input the dict representing one Pokemon, and computes the feature representation.
def calc_features(row):  
    pkmnFeatures = np.array([row['Attack'], row['Sp. Atk'], row['Speed'], row['Defense'], row['Sp. Def'], row['HP']]).astype(np.int64)
    return pkmnFeatures

# This function performs complete linkage hierarchical agglomerative clustering on the given pokemon, and returns a numpy array representing the clustering.
def hac(features):
    pokemonDict = {}
    currentTree = []
    indexList = []
    for i in range(len(features)):
        pokemonDict[i] = features[i]
        currentTree.append([i])
        indexList.append([i])

    hacArray = np.array(np.empty((len(features)-1, 4)))

    # First loop is to merge the tree until there is only one cluster
    for i in range(len(features)-1):
        # Second/Third loops are to find the closest two clusters/vertices
        currSmallestDistance = None
        currSmallestCluster1 = None
        currSmallestCluster2 = None
        for j in range(len(currentTree)):
            for k in range(len(currentTree)):
                if j != k:
                    # Fourth/Fifth loops are to use complete linkage
                    maxVal = None
                    for l in range(len(currentTree[j])):
                        for m in range(len(currentTree[k])):
                            currValue = np.linalg.norm(pokemonDict[currentTree[j][l]] - pokemonDict[currentTree[k][m]])
                            if maxVal == None or currValue > maxVal:
                                maxVal = currValue

                    # check if the current distance using complete linkage is the smallest
                    if currSmallestDistance == None or maxVal <= currSmallestDistance:

                        # sorting
                        if currSmallestDistance == maxVal:
                                # tiebreaker
                                if indexList.index(currentTree[j]) < indexList.index(currentTree[k]):
                                    if indexList.index(currentTree[j]) < indexList.index(currSmallestCluster1):
                                        currSmallestDistance = maxVal
                                        currSmallestCluster1 = currentTree[j]
                                        currSmallestCluster2 = currentTree[k]
                                    elif indexList.index(currentTree[j]) == indexList.index(currSmallestCluster1):
                                        if currentTree[k] < currSmallestCluster2:
                                            currSmallestDistance = maxVal
                                            currSmallestCluster1 = currentTree[j]
                                            currSmallestCluster2 = currentTree[k]
                                else:
                                    if indexList.index(currentTree[k]) < indexList.index(currSmallestCluster1):
                                        currSmallestDistance = maxVal
                                        currSmallestCluster1 = currentTree[k]
                                        currSmallestCluster2 = currentTree[j]
                                    elif indexList.index(currentTree[j]) == indexList.index(currSmallestCluster1):
                                        if indexList.index(currentTree[j]) < indexList.index(currSmallestCluster2):
                                            currSmallestDistance = maxVal
                                            currSmallestCluster1 = currentTree[k]
                                            currSmallestCluster2 = currentTree[j]
                        else:
                            if indexList.index(currentTree[j]) < indexList.index(currentTree[k]):
                                currSmallestDistance = maxVal
                                currSmallestCluster1 = currentTree[j]
                                currSmallestCluster2 = currentTree[k]
                            else:
                                currSmallestDistance = maxVal
                                currSmallestCluster1 = currentTree[k]
                                currSmallestCluster2 = currentTree[j]
        
        # Format the numpy array
        hacArray[i][0] = indexList.index(currSmallestCluster1)
        hacArray[i][1] = indexList.index(currSmallestCluster2)
        hacArray[i][2] = currSmallestDistance
        hacArray[i][3] = len(currSmallestCluster1) + len(currSmallestCluster2)
        
        # Merge the two clusters
        c1Index = currentTree.index(currSmallestCluster1)
        c2Index = currentTree.index(currSmallestCluster2)
        while len(currentTree[c2Index]) > 0:
            currentTree[c1Index].append(currentTree[c2Index].pop())
        indexList.append(currentTree[c1Index][:])
        currentTree.remove(currentTree[c2Index])

    return hacArray

# This function takes as input the output of hac, and plots the dendrogram.
def imshow_hac(Z):
    hierarchy.dendrogram(Z)
    plt.show()

# Demo of the program, using the first 20 Pokemon in the Pokedex.
imshow_hac(hac([calc_features(row) for row in load_data('Pokemon.csv')][:20]))