import numpy as np
import json

def print_dict(d: dict):
    class SetEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj,np.ndarray):
                return [list(o) for o in obj]

            return json.JSONEncoder.default(self, obj)
    print(json.dumps(d, cls = SetEncoder, sort_keys=True, indent=2))
    
def print_dict_schema(d : dict):
    def recurse_(d : dict, depth : int = 0):
        for _,k in enumerate(d):
            s = ("\t"*depth)+str(k)
            if type(d[k]) is dict:
                print(s)
            else:
                print(s+"\t: "+str(type(d[k])))
            
            if type(d[k]) is dict:
                recurse_(d[k], depth = depth + 1)

            if depth == 0:
                print("etc.")
                return
            
    return recurse_(d)

def split_fpn(mass : float):
    r = repr(mass)
    i,d = tuple(r.split("."))
    return (int(i),float("."+d))

def make_numpy_cloud(set_of_tuples : set):
    set_of_tuples = list(set_of_tuples)
    set_of_tuples = [ list(t) for t in set_of_tuples ]
    return np.asarray(set_of_tuples)

def make_point_cloud(
    ms : dict,
    transform : bool = False,
    log_x : bool = False,
    levels : int = 3,
):
    point_cloud = dict()
    for energy_level in range(levels):
        energy = "energy"+str(energy_level) # energy0, energy1, etc.
        if transform:
            point_cloud[energy] = make_numpy_cloud(set(
                [ split_fpn(m) for m in ms[energy]["mz"] ]
            ))
        else:
            point_cloud[energy] = make_numpy_cloud(set(
                [ t for t in zip(ms[energy]["mz"], ms[energy]["intens"])]
            ))
        #--- log of x-axis
        if log_x:
            point_cloud[energy][:,0] = np.log( point_cloud[energy][:,0] )
    #
    return point_cloud

def split_energy_levels(spectra : list, levels : int = 3):
    t = []
    for l in range(levels):
        t.append( [ ms["energy"+str(l)] for ms in spectra ] )
    
    return tuple(t)
