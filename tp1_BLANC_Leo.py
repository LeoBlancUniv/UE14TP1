import numpy as np
from math import pow
from random import randint
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def generateData(nbSample = 300, nbCenters = 4, nbFeatures = 2, random_state = 0):
    """
    this will generate data

    X reprensents the data

    """
    X, y = make_blobs(nbSample, nbCenters, nbFeatures, random_state)
    #print(np.shape(X))
    return X

def calcLq(x1, x2, q):

    """
    calculate and return the Lq norm over vector x1 and x2
    """

    ret = 0

    size = np.shape(x1)[0]
    #print(np.shape(x1), np.shape(x2))


    for i in range(size):
        #print(x1[i], x2[i])
        ret += abs(x1[i] - x2[i]) ** q

    ret = ret ** (1/q)

    return ret

def updateAssignation(dataMatrix, centroidsMatrix, q):
    """
    updates each data to the closest centroid based on Lq norm for distance calculation
    """

    ret = [0 for i in range(len(dataMatrix))]

    for i in range(len(dataMatrix)):

        dist = calcLq(dataMatrix[i], centroidsMatrix[0], q)  #init of dist

        for j in range(len(centroidsMatrix)):

            ndist = calcLq(dataMatrix[i], centroidsMatrix[j], q)

            if ndist < dist:
                dist = ndist
                ret[i] = j

        #at the end of that loop a[i] contains the updated centroid

    return ret


def updateCentroid(dataMatrix, assignation, nbCentroids):
    """
    updates all the centroids

    """

    k = nbCentroids

    d = len(dataMatrix[0])

    #print(k, d)

    ret = [ [ 0 for j in range(d) ] for i in range(k) ] #recreate an k*d array

    for i in range(nbCentroids):


        nbAdded = 0

        for j in range(len(dataMatrix)):

            if assignation[j] == i: #only if the data has been assigned to centroid i

                for l in range(d):
                    ret[i][l] += dataMatrix[j][l]  #we add the vector componant by componant
                    nbAdded += 1

        #here we added all assigned vector to the centroid so we devide by the number of vec added
        if nbAdded != 0:
            for l in range(d):
                ret[i][l] /= nbAdded

            nbAdded = 0


    return ret


def computeKmeansLoss(dataMatrix, centroidsMatrix, assignation, q):
    """
    return the loss function of the k means alg
    """

    ret = 0

    tmp = [0 for i in range(len(centroidsMatrix))]

    for i in range(len(assignation)):
        centroid = assignation[i]
        tmp[centroid] += calcLq(dataMatrix[i], centroidsMatrix[centroid], q)

    for dist in tmp:
        ret += dist

    return ret

def myKmeans(dataMatrix, nbCentroids, q, nbIteration):

    print("init")

    centroidsMatrix = []
    assignation = []
    lossArr = []

    #init centroidsMatrix

    usedArr = []

    j = 0

    while (j < nbCentroids):

        random = randint(0, len(dataMatrix)-1)

        if random not in usedArr:
            centroidsMatrix += [dataMatrix[random]]

            j += 1



    print("end of init")


    for i in range(nbIteration):
        #print("iteration nb :", i+1)
        assignation = updateAssignation(dataMatrix, centroidsMatrix, q)
        #print("updateAssignationassignation")
        centroidsMatrix = updateCentroid(dataMatrix, assignation, nbCentroids)
        #print("updateCentroid")
        lossArr += [computeKmeansLoss(dataMatrix, centroidsMatrix, assignation, q)]
        #print("computeKmeansLoss")


    print(lossArr)

    plt.scatter([x for x in range(nbIteration)], lossArr)

    plt.show()









if __name__ == "__main__":

    q = 2

    data = generateData()
    print(data)




    print("L"+ str(q) + " norm of the first 2 row of data :", calcLq(data[0], data[1], q))
    print("using function norm :",np.linalg.norm([data[0] - data[1]]))

    myKmeans(data, 4, 2, 20)



















#
