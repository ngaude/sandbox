import pandas as pd
import datetime
import time
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt

def epoch_sec(t):
    #?t string datetime in a 'YYYY-mm-dd HH:MM:SS' format
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

def resample_date(t, nt, sec = 900):
    #?t,nt string datetime in a 'YYYY-mm-dd HH:MM:SS' format
    # return range of datetime evenly spaced sec seconds between [t, nt[
    # return range of percentage [0,1[ of distance from t for zipped datetime
    s = epoch_sec(t)
    ns = epoch_sec(nt)
    d = ns - s
    if d<=0:
        return
    srange = range(int(math.ceil(s/sec)*sec),int(ns),sec)
    a = map(str_date,srange)
    b = map(lambda x: (x - s)/float(d),srange)
    return (a,b)

def resample_segment(t,x,y,r,nt,nx,ny,nr,sec = 900):
    # given a segment from x,y,r to nx,ny,nr between [t,nt[
    # return a list of st,sx,sy,sr 
    # with st evenly spaced sec seconds between [t, nt[
    (ds,ps) = resample_date(t,nt,sec)
    def interp(x,nx,p):
        return x+(nx-x)*p
    return [(d,interp(x,nx,p),interp(y,ny,p),interp(r,nr,p)) for d,p in zip(ds,ps)]

def randsample_position(x,y,r,n = 10):
    ls = np.random.uniform(0,r,n)
    ts = np.random.uniform(0,2*math.pi,n)
    return [ (x+l*math.cos(t),y+l*math.sin(t)) for l,t in zip(ls,ts)]

def flatten(l):
    # flatten a list of list
    return list(itertools.chain.from_iterable(l))

def randsample_segment(t,x,y,r,nt,nx,ny,nr,sec = 900, n = 10):
    # resample segment on sec seconds spaced position n times
    rss = resample_segment(t,x,y,r,nt,nx,ny,nr,sec)
    if not rss:
        # no sampled position, shortcut randsampling
        return []
    # identity sampling of time
    dsp = flatten([ [ad]*n for ad,ax,ay,ar in rss ])
    # random sampling of position
    xysp = flatten([ randsample_position(ax,ay,ar,n) for ad,ax,ay,ar in rss ])
    # zip time and position
    rsp = [(d,int(x),int(y)) for d,(x,y) in zip(dsp,xysp)]
    return rsp

def build_date_index(dxy):
    # build a date index dictionnary for [(d,x,y)] data description
    # sort by date and build date dict
    dxy.sort()
    didx = {}
    dcurr = dxy[0][0]
    icurr = 0
    for i,(d,x,y) in enumerate(dxy):
        if d > dcurr:
            # new date detected,add index in date dict
            didx[dcurr]=(icurr,i)
            dcurr = d
            icurr = i
    # add last index in date dict
    didx[dcurr] = (icurr,len(dxy))
    return didx


def density_map_from_date(dxy,didx,t):
    # dxy is the [(d,x,y)] data description
    # didx is the associated date index dictionnary
    # t string datetime in a 'YYYY-mm-dd HH:MM:SS' format 
    (d,x,y) = zip(*dxy[didx[t][0]:didx[t][1]])
    assert (not d) or (t==min(d) and t==max(d))
    plt.figure(t)
    bins = (range(550000,950000,2000),range(2050000,2550000,2000))
    plt.hist2d(x, y, bins=bins)
    plt.set_cmap('hot')
    plt.axis('off')
    #plt.colorbar()
    #plt.show(block=False)
    plt.savefig('density_map_'+str(t)+'.png')

# df expected fields : dat_heur_debt,x,y,r,ndat_heur_debt,nx,ny,nr
df = pd.read_csv('ngaude_paris_geneve_segment.tsv',sep = '\t')


# filter null duration segment
df = df[df.dat_heur_debt != df.ndat_heur_debt]

dxy = df.apply(lambda r: randsample_segment(r.dat_heur_debt,r.x,r.y,r.r,r.ndat_heur_debt,r.nx,r.ny,r.nr,sec = 900, n = 1), axis = 1)

# flatten the random sampled position
dxy = flatten(dxy)

# sort & build associated index
didx = build_date_index(dxy)

# map it
for i in range(0,80000,900):
    t = str_date(epoch_sec('2014-04-01 04:45:00') + i)
    density_map_from_date(dxy,didx,t)

# encode
# mencoder mf://image/*.png -mf w=320:h=240:fps=4:type=png -ovc lavc -lavcopts vcodec=mp4 -oac copy -o output.avi

