import pandas as pd

from aggregation_factory import  DataPack, AxisFunc, AxisPref, AggregationSolver

def solve_and_save(idf,idesc,fname):
    solver = AggregationSolver(DataPack(idf[idesc.keys()], idesc))
    solver.solve()
    if (solver.solution):
        odf = (solver.solution[0])[1].df
        cnil = (solver.solution[0])[1].cnil
        odf.to_csv(fname, sep = '\t', index = False)
        print fname,'cnil=',cnil,'row=',len(odf),'/',len(idf)
    return
    

#######################################################


idf = pd.read_csv('ngaude_lille_world_travellers_enriched.tsv', sep = '\t').fillna(0)
idesc = {'pays' : (AxisPref.mandatory, AxisFunc.aggregate_pays,0), 
        'home_iris' : (AxisPref.desired, AxisFunc.aggregate_iris,0),
        'work_iris' : (AxisPref.desired, AxisFunc.aggregate_iris,0),
        'dat_debt' :  (AxisPref.desired, AxisFunc.aggregate_date,4),
        'day_of_week' : (AxisPref.optional, AxisFunc.remove,0),
        'age_pers' : (AxisPref.optional, AxisFunc.remove,0),
        'scaling' : (AxisPref.individual, sum)
        }

solve_and_save(idf,idesc,'ngaude_lille_world_travellers_cnil.tsv')

#######################################################


idf = pd.read_csv('ngaude_lille_istambul_travellers_enriched.tsv', sep = '\t').fillna(0)
idesc = {'duration' : (AxisPref.optional, AxisFunc.aggregate_duration_day,0),
           'dat_debt' :  (AxisPref.desired, AxisFunc.aggregate_date,4),
           'day_of_week' : (AxisPref.optional, AxisFunc.remove,0),
           'home_iris' : (AxisPref.mandatory, AxisFunc.aggregate_iris,3),
           'work_iris' : (AxisPref.optional, AxisFunc.aggregate_iris,5),
           'age_pers' : (AxisPref.optional, AxisFunc.remove,0),
           'scaling' : (AxisPref.individual, sum)
           }

solve_and_save(idf,idesc,'ngaude_lille_istambul_travellers_cnil.tsv')

#######################################################

idf = pd.read_csv('ngaude_lille_istambul_hubbers_enriched.tsv', sep = '\t').fillna(0)
idesc = {'pays' : (AxisPref.mandatory, AxisFunc.aggregate_pays,0), 
           'duration' : (AxisPref.optional, AxisFunc.aggregate_duration_day,0),
           'dat_debt' :  (AxisPref.desired, AxisFunc.aggregate_date,4),
           'day_of_week' : (AxisPref.optional, AxisFunc.remove,0),
           'home_iris' : (AxisPref.optional, AxisFunc.aggregate_iris,3),
           'work_iris' : (AxisPref.optional, AxisFunc.aggregate_iris,3),
           'age_pers' : (AxisPref.optional, AxisFunc.remove,0),
           'scaling' : (AxisPref.individual, sum)
           }     

solve_and_save(idf,idesc,'ngaude_lille_istambul_hubbers_cnil.tsv')

#######################################################



idf = pd.read_csv('ngaude_lille_turkish_roamers_enriched.tsv', sep = '\t').fillna(0)
idesc = {'duration' : (AxisPref.optional, AxisFunc.aggregate_duration_day,0),
           'dat_debt' :  (AxisPref.desired, AxisFunc.aggregate_date,4),
           'day_of_week' : (AxisPref.optional, AxisFunc.remove,0),
           'scaling' : (AxisPref.individual, sum)
           }

solve_and_save(idf,idesc,'ngaude_lille_turkish_roamers_cnil.tsv')

#######################################################