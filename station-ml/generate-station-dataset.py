import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def cellid(x,y):
    x = max(0,min(int(x),399))
    y = max(0,min(int(y),399))
    return x+y*400

def home_cell(x,y):
    f = int(np.random.uniform(5,75))
    cxy = np.random.uniform(3,size=(f,2))-1.5
    #cxy = np.zeros(shape=(f,2))
    return (f,Counter([cellid(x+cx,y+cy) for cx,cy in cxy]))

n = 4000000

xf = open('xhome.dat', 'w')
yf = open('yhome.dat', 'w')

#y = np.random.uniform(size=(n,2))*400
#dataset = [ (round(hx,4),round(hy,4),home_cell(hx,hy)) for hx,hy in y]
#for i,e in enumerate(dataset):

for i in range(n):
    [hx,hy] = np.random.uniform(size=(2))*400
    e = (round(hx,3),round(hy,3),home_cell(hx,hy))
    xh = e[0]
    yh = e[1]
    w = e[2][0]
    cell = e[2][1]
    features = ' |influenceFeatures '
    features += ' '.join([str(k)+':'+str(1.*v/w) for k,v in cell.iteritems()])
    xf.write(str(xh)+features+'\n')
    yf.write(str(yh)+features+'\n')
    if (i%100000==0):
        print i,'/',n

xf.close()
yf.close()

"""

rm x_train.dat.cache
head -n 2000000 xhome.dat > x_train.dat
tail -n 2000000 xhome.dat > x_test.dat

rm y_train.dat.cache
head -n 2000000 yhome.dat > y_train.dat
tail -n 2000000 yhome.dat > y_test.dat


vw -d x_train.dat -f x_predictor.vw -c --passes 100 --loss_function squared --bfgs
vw -d x_test.dat -t -i x_predictor.vw -p x_predictions.txt

vw -d y_train.dat -f y_predictor.vw -c --passes 100 --loss_function squared --bfgs
vw -d y_test.dat -t -i y_predictor.vw -p y_predictions.txt

"""

e = np.zeros(2000000)

xt = open('x_test.dat', 'r')
xp = open('x_predictions.txt', 'r')
yt = open('y_test.dat', 'r')
yp = open('y_predictions.txt', 'r')

for i in range(2000000):
    ax = float(xt.readline().split(' ')[0])
    ay = float(yt.readline().split(' ')[0])
    bx = float(xp.readline().strip())
    by = float(yp.readline().strip())
    e[i] = math.sqrt(math.pow(ax-bx,2)+math.pow(ay-by,2))

xt.close()
xp.close()
yt.close()
yp.close()

plt.hist(e,bins=1000,range=(0,2))
plt.show()



#MINI TEST, CAN'T BELIEVE IT !!!!#
#MINI TEST, CAN'T BELIEVE IT !!!!#
#MINI TEST, CAN'T BELIEVE IT !!!!#
#MINI TEST, CAN'T BELIEVE IT !!!!#
n = 45678
xf = open('xminitest.dat', 'w')
yf = open('yminitest.dat', 'w')

for i in range(n):
    [hx,hy] = np.random.uniform(size=(2))*400
    e = (round(hx,3),round(hy,3),home_cell(hx,hy))
    xh = e[0]
    yh = e[1]
    w = e[2][0]
    cell = e[2][1]
    features = ' |influenceFeatures '
    features += ' '.join([str(k)+':'+str(1.*v/w) for k,v in cell.iteritems()])
    xf.write(str(-xh)+features+'\n')
    yf.write(str(-yh)+features+'\n')
    if (i%10000==0):
        print i,'/',n

xf.close()
yf.close()

"""
vw -d xminitest.dat -t -i x_predictor.vw -p x_minipredictions.txt
vw -d yminitest.dat -t -i y_predictor.vw -p y_minipredictions.txt
"""


