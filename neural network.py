from scipy import misc
import numpy as np
from math import exp

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + exp(gamma))
    return 1 / (1 + exp(-gamma))

def trainNN(datanupx,datanupy):
    a = np.ones(datanupx.shape[0])
    datanupx = np.insert(datanupx, 0, values=a, axis=1) #184*961
    datanupx = datanupx/255.0
    datanupw = np.random.random_integers(-1, 1, (961, 100))  # 961*100
    datanupw2 = np.random.random_integers(-1, 1, (101, 1))  # 101*1
    for i in range(0, 1000):
        # t = np.random.randint(0,datanupx.shape[0],1)
        # randomx = datanupx[t] #1*961
        # y = datanupy[t]
        # y = float(y)
        # for j in range(0,datanupx.shape[0]):
        t = np.random.randint(0, datanupx.shape[0], 1)
        randomx = datanupx[t]  # 1*961
        y = datanupy[t]
        y = float(y)
        datanup_s = (randomx * datanupw).T #1*100
        datanups = np.matrix(map(sigmoid,datanup_s)) #1*100
        b = np.ones(1)
        datanupx2 = np.insert(datanups, 0, values=b, axis=1) #1*101
        datanup_s2 = datanupx2 * datanupw2 #1*1
        datanupx3 = sigmoid(datanup_s2)
        g1 = 2.0 *(datanupx3 - y)* datanupx3 * (1.0 - datanupx3)
        w2 = datanupw2 - 0.1 * g1 * datanupx2.T #101*1 #
        a = np.multiply(datanupx2,(1.0 - datanupx2))#1*101*1*101, multiply not dot
        b = np.dot(w2,g1)#101*1
        g2 =np.matrix(np.multiply(a.T,b)) #101*1
        g2 = np.delete(g2,0,0) #100*1
        # g2 = np.dot((datanupx2.T,(1.0 - datanupx2)), np.dot(w2,g1)) #101*1 ---
        c = g2 * randomx#100*961
        d = np.dot(0.5,c)
        w = datanupw - d.T#961*100
        datanupw2 = w2
        datanupw = w
    return w,w2

def testNN(datanupx,datanupy,w,w2):
    l = 0
    prediction = []
    a = np.ones(datanupx.shape[0])
    datanupx = np.insert(datanupx, 0, values=a, axis=1)  # 83*961
    datanupx = datanupx / 255.0
    for i in range(0, datanupy2.shape[0]):
        siglex = datanupx[i,:] #1*961
        sigley = datanupy[i,0]
        sigley = float(sigley)
        datanup_s = np.dot(siglex,w) # 1*100
        v = datanup_s
        datanups = np.matrix(map(sigmoid, datanup_s.T))  # 1*100
        b = np.ones(1)
        datanupx2 = np.insert(datanups, 0, values=b, axis=1)  # 1*101
        datanup_s2 = datanupx2 * w2  # 1*1
        datanupx3 = sigmoid(datanup_s2)
        if datanupx3 >= 0.5:
            datanupx3 = 1.0
            prediction.append("True")
        else:
            datanupx3 = 0.0
            prediction.append("False")
        if datanupx3 == sigley:
            l += 1.0
    accuracy = l/datanupx.shape[0]
    print "prediction_list = ", prediction
    print "accuracy = ", accuracy

f = open("downgesture_train.list")
datanup_x = []
datanup_y = []
for line in f.readlines():
    line = line.strip()
    if "down" in line:
        datanup_y.append(1.0)
    else:
        datanup_y.append(0.0)
    c = misc.imread(line)
    m = c.shape[0]*c.shape[1]
    x = list(c.reshape(1,m))
    datanup_x += x
datanupx = np.matrix(datanup_x) #184*960
datanupy = (np.matrix(datanup_y)).T #184*1
w,w2 = trainNN(datanupx,datanupy)

f2 = open("downgesture_test.list")
datanup_x2 = []
datanup_y2 = []
for line in f2.readlines():
    line = line.strip()
    if "down" in line:
        datanup_y2.append(1)
    else:
        datanup_y2.append(0)
    c2 = misc.imread(line)
    m2 = c2.shape[0]*c2.shape[1]
    x2 = list(c2.reshape(1,m2))
    datanup_x2 += x2
datanupx2 = np.matrix(datanup_x2) #83*960
datanupy2 = (np.matrix(datanup_y2)).T #83*1
testNN(datanupx2,datanupy2,w,w2)






