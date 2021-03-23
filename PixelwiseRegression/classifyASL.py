#Index = [0, 4, 5, 6]
#Mid   = [0, 7, 8, 9]
#Ring  = [0, 10, 11, 12]
#Small = [0, 13, 14, 15]
#Thumb = [0, 1, 2, 3]
#Fingers = [0=Thumb, 1=Index, 2=Mid, 3=Ring, 4=Small]

#Radiusberechnung:
#1. Welche Finger sind zu 100% angewinkelt: Mindestradius
#1.1 Daumenspitze ist ausgetreckt weiter entfernt als Zeigefinger1. Wert
#2. Berechne Abstand Handfläche zu next joints von (bisher) nicht angewinkelten Fingern
#3. Checke welche Fingerkuppelgelenke innerhalb des Handflächenradius ist


#Next:
#4. Finger die sich "berühren"

#Example
#x1(300, 350)
#x2(350, 500)

#x = 50
#y = 150
#x²+y²=d²
#-> d= 158, 145, ... -> 140

import numpy as np
import math
from test_samples import draw_skeleton

def calcPalm(results):
    #TODO Mindestradius
    indexList = [2,4,7,10,13]
    distances = []
    for index in indexList:
        distances.append(calcDistance(results[0], results[index]))
    return sum(distances)/len(distances)
    
#2 tuples Inputtype:(x,y)
def calcDistance(x1, x2):
    #print("x1", x1)
    #print("x2", x2)
    xDev = abs(x1[0]-x2[0])
    yDev = abs(x1[1]-x2[1])
    #pythagoras
    
    return math.sqrt(xDev**2 + yDev**2)

#check if joint if fingertip is inside the radius
def fingerStretched(results, palmRadius):
    #Booleanlist, True if fingers are stretched: 0=Thump,....4=small finger
    fingers = [False, False, False, False, False]
    fingerLength=0 
    indexList = [3, 6, 9, 12, 15]
    root = results[0]
    for index in indexList:
        if index == 3:
            fingerLength = 0
        elif index == 15:
            fingerLength = palmRadius*(6/10)
        else:
            fingerLength = palmRadius*(2/3)
        
        distanceToPalm = calcDistance(results[0], results[index])
        distanceRootToTip = calcDistance(results[index], results[index-2])
        if distanceToPalm > palmRadius:
            fingers[int(index/3-1)] = True
            
            if distanceRootToTip < fingerLength:
                print("Extracase triggered")
                fingers[int(index/3-1)] = False
        else:
            fingers[int(index/3-1)] = False
    print(fingers)
    return fingers

def classifyHandSign(fingers):
    #Zeigefinger -> 9
    if fingers == [False, False, True, True, True]:
        return 9
    # 
    if fingers.count(True)==5:
        return 5
    if fingers == [True, True, True, False, False]:
        return 3
    # Thumb always 0 from here on
    if fingers[1:5].count(True)==4 and fingers[0] == False:
        return 4
    if fingers == [False, True, False, False, False]:
        return 1
    if fingers == [False, True, True, False, False]:
        return 2
    if fingers == [False, True, True, True, False]:
        return 6
    if fingers == [False, True, True, False, True]:
        return 7
    if fingers == [False, True, False, True, True]:
        return 8
    if fingers == [True, True, False, False, True]:
        return "i love you"
    else:
        return -1

'''

#Denormalize the Uv coordinates
joints = joints * (512 - 1) + np.array([512 // 2, 512 // 2])
print(joints)
#convert to xy or int? coordinates
_joint = [(int(joints[i][0]), int(joints[i][1])) for i in range(joints.shape[0])]

palmRadius = calcPalm(_joint)
fingers = fingerStretched(_joint, palmRadius)
prediction = classifyHandSign(fingers)
print(prediction)'''














