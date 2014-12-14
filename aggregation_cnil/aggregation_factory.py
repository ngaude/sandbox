import pandas as pd
import math
import heapq

# enumeration of axis preferences
class AxisPref:
    mandatory = 1
    desired = 2
    optional = 3
    value = 4
    individual = 5    

class DataPack:
    def __init__(self, df = pd.DataFrame(), desc = {},level = {}):
        '''
        DataPack instance object contains all the data you need
        '''
        # data frame 
        self.df = df
        # description of dataframe columns type
        self.desc = desc
        # remove duplicate and tidy up the dataframe
        self.__remove_duplicate()
        # compute data pack entropy
        self.entropy = self.__entropy()
        # compute the cnil rate
        self.cnil = self.__cnil()
        return
        
    def desc_by_pref(self):
        '''
        reverse a datapack axis dictionnary enumerate by preferences
        '''
        d = {AxisPref.mandatory:[], AxisPref.desired:[], AxisPref.optional:[]}
        for axis, value in self.desc.iteritems():
            pref = value[0]
            d.setdefault(pref,[]).append(axis)
        return d
        

    def desc_inc_level(self,axis):
        '''
        increase level associated to a given axis
        '''        
        assert axis in self.desc.keys()
        assert axis in (AxisPref.mandatory, AxisPref.desired, AxisPref.optional)
        t = self.desc[axis]
        self.desc[axis] = (t[0], t[1], t[2]+1)
        return
    
    def __cnil(self):
        cnil_threshold = 11
        ind = self.desc_by_pref()[AxisPref.individual][0]
        ind_sum = self.df[ind].sum()
        ind_compliant = self.df[ind][self.df[ind] >= cnil_threshold].sum()
        return float(ind_compliant)/float(ind_sum)

    def __entropy(self):
        def vec_entropy(seq):
            '''
            compute the series entropy 
            given series seq as a discrete rv
            '''
            pk = seq.value_counts()/len(seq)
            s = - sum(pk * map(math.log, pk))
            return s
        pref = self.desc_by_pref()
        # compute mandatory columns entropy
        m = sum([ vec_entropy(s) for c,s in self.df[pref[AxisPref.mandatory]].iteritems()])
        # compute desired columns entropy
        d = sum([ vec_entropy(s) for c,s in self.df[pref[AxisPref.desired]].iteritems()])
        # compute optional columns entropy
        o = sum([ vec_entropy(s) for c,s in self.df[pref[AxisPref.optional]].iteritems()])
        # to promote the mandatory vs desired vs optional, 
        # I choose to apply some fibonacci coeff
        return 21 * m + 13 * d + 8 * o
        
    def __remove_duplicate(self):
        d = self.desc_by_pref()
        # check existence of at least an individual axis
        assert d.get(AxisPref.individual,False)

        # build aggregating funcs dict for individual and value axis list
        funcs = {col : self.desc[col][1] for col in d[AxisPref.individual] + d.get(AxisPref.value,[])}
        
        # add to  list all other required axis
        cols = d.get(AxisPref.mandatory,[]) + d.get(AxisPref.desired,[]) + d.get(AxisPref.optional,[])

        self.df = self.df.groupby(cols).agg(funcs).reset_index()
        return

class AxisFunc:
    
    # preload a continent-pays association needed for the 
    __dfcont = pd.read_csv('ngaude_pays_continent.tsv', sep = '\t').fillna(0)

    @staticmethod
    def aggregate(dp, axis):
        '''
        return a datapack with an higher level of aggregation for 'axis'
        '''
        # check the 'axis' column exists in description and dataframe
        assert axis in dp.desc.keys()        
        assert axis in list(dp.df.columns)
        # check the 'axis' column can be aggregated
        assert dp.desc[axis][0] in (AxisPref.mandatory, AxisPref.desired, AxisPref.optional)
        #call the axis associated function
        ind = dp.desc_by_pref()[AxisPref.individual][0]
        func = dp.desc[axis][1]
        level = dp.desc[axis][2]
        dp =  func(dp, axis, ind, level)
        # if axis has not been removed 
        # then increment the next aggregation level
        if axis in dp.desc.keys():
            t = dp.desc[axis]
            dp.desc[axis] = (t[0], t[1], t[2]+1)
        return dp
        
    @staticmethod
    def remove(dp, axis, ind, level):
        '''
        return a datapack without 'axis'
        '''
        # build a dataframe without axis column
        cols = list(dp.df.columns)
        cols.remove(axis)        
        df = dp.df[cols].copy()
        # build an associated description without axis
        desc = dp.desc.copy()
        desc.pop(axis, None)
        # build a new a datapack
        return DataPack(df,desc)   


    @classmethod
    def aggregate_pays(cls, dp, axis, ind, level):
        '''
        partially aggregate an axis of type 'pays'
        '''        

        threshold = [2,4,8,16,32,64,128]
        if (level >= len(threshold)):
            return cls.remove(dp, axis, ind, 0) 
        # add a continent comprehension to dataframe
        df = dp.df.merge(cls.__dfcont, how='inner', left_on=axis, right_on='pays')
        # select rows below the given threshold         
        filt = (df[ind] <= threshold[level])
        # where the partial aggregation request it
        # put a 'continent' value in 'pays' field 
        df.ix[filt, [axis]] = df[filt].continent
        #return the new datapack
        return DataPack(df,dp.desc.copy())
    
    @classmethod
    def aggregate_iris(cls,dp, axis, ind, level):
        '''
        partially aggregate an axis of type 'iris'
        
        Any '9char' iris string can be partially aggregated 
        to the commune (75001) or the departement (75) it belongs to
        '''

        threshold = [2,4,8,16,32,64,128]
        if (level >= len(threshold)):
            return cls.remove(dp, axis, ind, 0) 

        def convert(iris):
            if len(iris)==9:
                return iris[:5]
            if len(iris)==5:
                return iris[:2]
            return iris
            
        df = dp.df.copy()
        # select rows below the given threshold 
        filt = (df[ind] <= threshold[level])
        # where the partial aggregation request it
        # put a 'commune' or 'departement' value in 'iris' field         
        df.ix[filt, [axis]] = df[filt][axis].astype(str).apply(convert)
        #return the new datapack
        return DataPack(df,dp.desc.copy())
    
    @classmethod
    def aggregate_date(cls, dp, axis, ind, level):
        '''
        completely aggregate an axis of type 'date'
        
        level 0 : => YYYY-MM-DD HH:mm
        level 1 : => YYYY-MM-DD HH
        level 2 : => YYYY-MM-DD morning|afternoon|evening|night
        level 3 : => YYYY-MM-DD
        level 4 : => YYYY-MM
        level 5 : => YYYY spring|summer|fall|winter
        level 6 : => YYYY
        level 7 : axis removal
        '''

        msc = [16,13,10,10,7,4,4]
        if (level >= len(msc)):
            return cls.remove(dp, axis, ind, 0)
        
        def convert(date,level):
            ndate = date[:msc[level]]

            if (level==2):
                day_period = ['night','morning','afternoon','evening']
                ndate += ' ' + day_period[int(date[11:13])/6]
            elif (level==5):
                year_period = ['spring','summer','fall','winter']
                ndate += ' ' + year_period[int(date[5:7])/6]
                
            return ndate
            
        df = dp.df.copy()
        # modify date for the current level using convert function
        df[axis] = df[axis].astype(str).apply(lambda d : convert(d,level))
        #return the new datapack
        return DataPack(df,dp.desc.copy())

    @classmethod
    def aggregate_duration_day(cls, dp, axis, ind, level):  
        if (level == 0):
            # convert duration in seconds for the first level
            dp.df[axis] = dp.df[axis].apply(lambda h:int(86400*h))
        return cls.aggregate_duration_sec(dp, axis, ind, level)

    @classmethod
    def aggregate_duration_hour(cls, dp, axis, ind, level):  
        if (level == 0):
            # convert duration in seconds for the first level
            dp.df[axis] = dp.df[axis].apply(lambda h:int(3600*h))
        return cls.aggregate_duration_sec(dp, axis, ind, level)    
    
    @classmethod
    def aggregate_duration_sec(cls, dp, axis, ind, level):
        '''
        completely aggregate an axis of type 'duration' 
        expressed in seconds in input dataframe for first level
        the aggregation progress for the finest grain duration (5mn) to the largest (30j)
        '''
        
        sec_range = [300,900,3600,7200,14400,43200,86400,172800,604800,1209600,2592200]
        str_range = ['5mn','15mn','1h','2h','4h','12h','1j','2j','7j','14j','30j']

        if (level >= len(sec_range)):
            return cls.remove(dp, axis, ind, 0)       
        
        def convert(dur,level):
            if level == 0:
                # assume date is integer given in seconds
                dur = int(dur)
                if dur >= sec_range[-1]:
                    return '>'+str_range[-1]
                if dur <= sec_range[0]:
                    return '<'+str_range[0]
                # fixme, use dichotomic search given a sorted array
                for i,sec in enumerate(sec_range[1:]):
                    if dur < sec:
                        return str_range[i-1]+'-'+str_range[i]
            else:
                # assume date is string given acccording to the syntax
                # 5mn-15mn , 15mn-h,....
                if (dur[0]=='>'):
                    return dur
                if (dur[0]=='<'):
                    return '<'+str_range[level]
                lower = dur.split('-')[0]
                if lower in str_range[level:]:
                    return dur
                return '<'+str_range[level]
        
        df = dp.df.copy()
        # modify date for the current level using convert function
        df[axis] = df[axis].apply(lambda d : convert(d,level))
        #return the new datapack
        return DataPack(df,dp.desc.copy())

class AggregationSolver:
    cnil_entropy_ratio = 3.0
    cnil_acceptance = 0.95
    def __init__(self,dp):
        '''
        start a new aggregation solving path finding given a root datapack
        '''
        # visited combination of aggregation function
        self.visited = []
        # root aggregation solution
        self.root = dp
        # best aggregation solution found for now
        self.solution = []        
        # priority queue composed of element tuple as :
        # (priority, datapack,aggregation-path as a list of axis)
        self.search = []
        heapq.heappush( self.search, ( self.__priority(dp), dp, []) )        
        return

    def __priority(self, dp):
        '''
        priority is a combination of the entropy loss and the cnil uncompliance
        '''
        return - AggregationSolver.cnil_entropy_ratio * dp.cnil  - (dp.entropy / self.root.entropy)
    
    def solve(self):
        i = 0
        while not self.solution and i<1000:
            i+=1
            self.__iterate()
    
    def __iterate(self):

        # fixme,I shall limit the size of the priority queue 
        # for obvious memory space reason

        (prio, dp,path) = heapq.heappop(self.search)        
        d = dp.desc_by_pref()
        cols = d.get(AxisPref.mandatory,[]) + d.get(AxisPref.desired,[]) + d.get(AxisPref.optional,[])
        
        for axis in cols:
                # check if axislevel to explored has alreday been visited
                axislevel = (axis,dp.desc[axis][2])
                npath = path+[axislevel]
                combination =  set(path)
                combination.add(axislevel)
#                print '---------'
#                print 'path = ',path
#                print 'axislevel = ',axislevel
#                print 'npath = ',npath
#                print 'combination = ',combination
                # do not visit this aggregation path twice
                if combination in self.visited:
                    continue
                
                ndp = AxisFunc.aggregate(dp,axis)
                nprio = self.__priority(ndp)
                npath = path + [axislevel]
                if (ndp.cnil >= AggregationSolver.cnil_acceptance):
                    heapq.heappush(self.solution, ( ndp.cnil, ndp, npath) )
                else:
                    heapq.heappush( self.search, ( nprio, ndp, npath) )
                    
                # mark this path of axislevel aggregation combination 
                # as a visited combination
                self.visited.append(combination)
        return
