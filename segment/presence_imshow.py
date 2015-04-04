import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time,datetime
import math
import sys


def epoch_sec(t):
    #t string datetime in a 'YYYY-mm-dd HH:MM:SS' format
    # return elapsed seconds from epoch time
    dt = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    s = time.mktime(dt.timetuple())
    return s

def str_date(s):
    # s elapsed seconds from epoch time
    # return string datetime in a 'YYYY-mm-dd HH:MM:SS' format
    dt = datetime.datetime.fromtimestamp(s)
    sdt = dt.strftime('%Y-%m-%d %H:%M:%S')
    return sdt 

def save_presence(t):
    m = ms[t]
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(m,vmin=vmin,vmax=vmax,interpolation='bicubic')
    fn = '/home/ngaude/workspace/data/image/presence_'+t+'.png'
    plt.savefig(fn,dpi = 720)
    plt.close()

"""
*interpolation*:
Acceptable values are *None*, 'none', 'nearest', 'bilinear',
'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
'bessel', 'mitchell', 'sinc', 'lanczos'
"""

def plot_presence(t):
    m = ms[t]
    plt.figure(t)
    plt.imshow(m,vmin=vmin,vmax=vmax,interpolation='bicubic')
    plt.show()

df = pd.read_csv('presence.csv', names = ('dat_heur','xmin','ymin','xmax','ymax','value'))

ms = df.groupby(df.dat_heur).value.apply(np.array)
xs = sorted(set(df.xmin))
ys = sorted(set(df.ymin))
ts = sorted(set(df.dat_heur))
w = len(xs)
h = len(ys)
for m in ms:
    m.shape =  (w,h)
    
# compute background map from presence maps
bkg = ms.mean()

# remove background map from presence maps
ms = [m-bkg for m in ms]
vmin = min([m.min() for m in ms])
vmax = max([m.max() for m in ms])

if len(sys.argv)>1:
    try:
        t = int(sys.argv[1])
    except ValueError:
        t = sys.argv[1]
    plot_presence(t)
else:    
    for i,t in enumerate(ts):
        print 'image '+str(i)+'/'+str(len(ts))+':',t
        save_presence(t)
