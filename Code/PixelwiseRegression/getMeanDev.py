import os
import math
import numpy as np 


#String to number floats, delete depth
def preprocessingOfList(lImgPreds):
    predsUV = np.array([[100,100,100]])
    for item in range(15):
        x = [float(lImgPreds[item*3]),float(lImgPreds[item*3+1]), float(lImgPreds[item*3+2])]
        predsUV = np.append(predsUV, [x], axis = 0)
        #d value gets lost
        #predsUV[i][i+2] = 
    predsUV = np.delete(predsUV, (0), axis=0)
    
    #convert to xy or int? coordinates
    #_joint44 = [(int(predsUV[i][0]), int(predsUV[i][1])) for i in range(predsUV.shape[0])]
    return predsUV
    
def distanceCalc3D(predsXYD,gtXYD):
    a1 = predsXYD[0]-gtXYD[0]
    a2 = predsXYD[1]-gtXYD[1]
    a3 = predsXYD[2]-gtXYD[2]
    
    return math.sqrt(a1**2+a2**2+a3**2)
    
#calc the x,y,z values of one image
def calcOne(sLabel,toggleGRTruth):
    lLabel = []
    lImgPreds = []
    if toggleGRTruth:
        sLabel = sLabel[:-1]
        lLabel = sLabel.split(" ")
        lLabel = lLabel[1:]
        lImgPreds = lLabel
    else:
        sImgPreds = sLabel
        sImgPreds = sImgPreds[:-1]
        lImgPreds = sImgPreds.split(" ")

    #Prediction as list
    sImgPreds = lLabel

    _joint44 = preprocessingOfList(lImgPreds)
    return _joint44

if __name__ == '__main__':
    
    lMeanError = []
    test = open("Data/ICVL/test.txt", "r")
    testLines = test.readlines()
    
    pred = open("Result/ICVL_default.txt", "r")
    testPredLines = pred.readlines()
    i=0
    for i in range(len(testLines)):
        predJoints = calcOne(testPredLines[i], False)
        gtJoints = calcOne(testLines[i], True)
        Error3D = []
        for i in range(len(predJoints)):
            Error3D.append(distanceCalc3D(predJoints[i],gtJoints[i]))
        lMeanError.append(sum(Error3D)/len(Error3D))
        #print(lMeanError[i])
    meanError = sum(Error3D)/len(Error3D)
    print("Final 3D mean error: ", meanError)
    
    test.close()
    pred.close()
