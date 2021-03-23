import os
import math



if __name__ == '__main__':
    
    sdImage = []
    
    test = open("Data/ICVL/test.txt", "r")
    testLines = test.readlines()
    
    pred = open("Result/ICVL_default.txt", "r")
    testPredLines = pred.readlines()
    i=0
    labels = []
    for i in range(len(testLines)):
        #Model's predictions convert from string to array
        sImgPreds = testPredLines[i]
        #Remove line break
        sImgPreds = sImgPreds[:-1]
        lImgPreds = sImgPreds.split(" ")
        
        sGrTruth = testLines[i]
        sGrTruth = sGrTruth[:-1]
        lGrTruth = sGrTruth.split(" ")
        lGrTruth = lGrTruth[1:]
        sumJoints = 0
        #calculate standard deviation
        for i in range(len(lImgPreds)):
            #print( lImgPreds[i] + lGrTruth[i])
            sumJoints = sumJoints + ((float(lImgPreds[i])-float(lGrTruth[i]))**2)
        if len(lImgPreds) != 48:
            print(len(lImgPreds))
        sd = math.sqrt(sumJoints/len(lImgPreds))
        sdImage.append(sd)
        print(sd)
    print(sdImage)
    mean = sum(sdImage)/len(sdImage)
    print(mean)
        
    test.close()
