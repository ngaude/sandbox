import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time,datetime
import math


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

def save_presence(d):
    m = np.array(df[df.dat_heur == d].value)
    xaxis = sorted(set(df.xmin))
    yaxis = sorted(set(df.ymin))
    w = len(xaxis)
    h = len(yaxis)
    print d,':',w,'x',h
    m.shape = (w,h)
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.contour(m)
    ax.imshow(m,vmin=0,vmax=vmax)
    fn = '/home/ngaude/workspace/data/image/presence_'+d+'.png'
    plt.savefig(fn,dpi = 720)
    plt.close()


def plot_presence(d):
    m = np.array(df[df.dat_heur == d].value)
    print m
    xaxis = sorted(set(df.xmin))
    yaxis = sorted(set(df.ymin))
    w = len(xaxis)
    h = len(yaxis)
    m.shape = (w,h)
    plt.figure(d)
    #plt.axis([min(xaxis),max(xaxis),min(yaxis),max(yaxis)])
    plt.imshow(m,vmin=0,vmax=vmax)
    plt.contour(m)
    plt.show()


vmax = max(df.value)*0.9

df = pd.read_csv('presence.csv', names = ('dat_heur','xmin','ymin','xmax','ymax','value'))
d = "1975-11-15 19:30:00"
m = np.array(df[df.dat_heur == d].value)
plot_presence(d)

for d in sorted(set(df.dat_heur)):
    if d[11:] != '04:30:00':
        save_presence(d)

