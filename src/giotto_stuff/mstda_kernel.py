from gtda.homology import EuclideanCechPersistence as Persistence
from gtda.diagrams import PersistenceEntropy
from gtda.pipeline import Pipeline
from gtda.diagrams import Filtering

from mstda_utils import make_gaussian_mixtures

import numpy as np
from functools import partial
import json
import os.path
import scipy
from multiprocessing import Pool, cpu_count
from functools import partial

class FeatureSpace:
    def __init__(
        self,
        json_file           : str   = None,
        energy_levels       : int   = 3,
        diagram_epsilon     : float = 0.01,
        homology_dimensions : list  = [0,1,2],
        normalized_entropy  : bool  = False,
        n_jobs              : int   = -1,
        n_samples           : int   = 50,
    ):
        if json_file == None:
            return
        
        assert type(json_file) is str, "json_file argument should be a path"
        assert os.path.exists(json_file), json_file + " missing?"
        assert json_file.lower().endswith(".json"), json_file + " is not a .json file?"
        
        with open(json_file, mode = "r") as data:
            self.db = json.load(data)

        self.energy_levels       = energy_levels
        self.diagram_epsilon     = diagram_epsilon
        self.homology_dimensions = homology_dimensions
        self.normalized_entropy  = normalized_entropy
        self.n_jobs              = cpu_count() - 1 if n_jobs == -1 else n_jobs
        self.n_samples           = n_samples
        
        self.pipeline  = Pipeline([
            ('diagram', Persistence(
                homology_dimensions = self.homology_dimensions,
                n_jobs              = self.n_jobs
            )),
            ('filter', Filtering(
                epsilon             = self.diagram_epsilon
            )),
            ('entropy', PersistenceEntropy(
                normalize           = self.normalized_entropy,
                n_jobs              = self.n_jobs
            )),
        ])
        (self.features, self.molecules) = self._compute_features(self.db)
    
    # return the fragments of a training SMILES
    def get_frags(
        self,
        SMILES
    ):
        return np.asarray(self.db[SMILES]["frag"])
    
    # return the feature vector of a training SMILES
    # to get all feature vectors, use self.features
    def get_feature_vector(
        self,
        SMILES : str
    ):
        try:
            idx = self.molecules.index(SMILES)
            return self.features[idx, :]
        except ValueError as _:
            return None
                
    def __str__(self) -> str:
        if hasattr(self, "features"):
            return "\n".join([
                "Feature space:",
                str("Molecules:\t\t\t")                 + str(len(self.molecules)),
                str("Topological features:\t\t")        + str(self.features.shape),
                str("Homology dimensions:\t\t")         + str(self.homology_dimensions),
                str("Energy levels:\t\t\t")             + str(self.energy_levels),
                str("Mass transformed:\t\t"             + str(self.transform)),
                str("Logarithmic integer mass:\t"       + str(self.log_x)),
                str("Persistence diagram epsilon:\t"    + str(self.diagram_epsilon)),
                str("Persistence entropy normalized:\t" + str(self.normalized_entropy)),
                str("Parallelism (n_jobs):\t\t")        + str(self.n_jobs)
            ]).expandtabs()
        else:
            return "Empty space"
        
    @classmethod
    def load(cls, save_path : str):
        assert type(save_path) is str, "save_path should be a path"
        assert os.path.exists(save_path), save_path + " missing?"
        
        fspace = FeatureSpace()
        with open(save_path, mode = "r") as j:
            save_dict = json.load(j)
        
        for s,setting in save_dict["settings"].items():
            setattr(fspace, s, setting)
        
        fspace.features = np.asarray(save_dict["features"])
        fspace.molecules = save_dict["molecules"]
        fspace.pipeline = Pipeline([
            ('diagram', Persistence(
                homology_dimensions = fspace.homology_dimensions,
                n_jobs              = fspace.n_jobs
            )),
            ('filter', Filtering(
                epsilon             = fspace.diagram_epsilon
            )),
            ('entropy', PersistenceEntropy(
                normalize           = fspace.normalized_entropy,
                n_jobs              = fspace.n_jobs
            )),
        ])
        fspace.db     = save_dict["db"]
        return fspace
    
    def save(self, save_path : str):
        save_dict = dict()
        save_dict["settings"] = dict()
        save_dict["settings"]["energy_levels"]       = self.energy_levels
        save_dict["settings"]["diagram_epsilon"]     = self.diagram_epsilon
        save_dict["settings"]["homology_dimensions"] = self.homology_dimensions
        save_dict["settings"]["normalized_entropy"]  = self.normalized_entropy
        save_dict["settings"]["n_jobs"]              = self.n_jobs        
        
        save_dict["features"]  = [ list(f) for f in self.features ]
        save_dict["molecules"] = self.molecules
        save_dict["db"]        = self.db
        
        with open(save_path, mode = "w") as j:
            json.dump(save_dict, j)
        
        print("saved feature space to: " + os.path.abspath(save_path))
    
    def _make_point_clouds(
        self,
        mass_specs : dict,
        raw        : bool = False,
    ) -> dict :
        gm = {
            k : make_gaussian_mixtures(
                tuple([ v["energy"+str(e)] for e in range(self.energy_levels) ]),
            ) for k, v in mass_specs.items()
        }
        for m in list(gm.keys()):
            max_mass = max([ max( self.db[m]["energy"+str(e)]["mz"] ) for e in range(self.energy_levels) ])
            x_def = np.linspace(0, max_mass, num = self.n_samples)
            tmp = np.zeros((self.n_samples, self.energy_levels))
            for e in range(self.energy_levels):
                tmp[:,e] = gm[m][e](x_def)
            gm[m] = tmp
            
        return gm
    
    def _compute_features(
        self,
        mass_specs : dict
    ) -> tuple :
        point_clouds = self._make_point_clouds(mass_specs)
        return (
            self.pipeline.fit_transform(list(point_clouds.values())),
            list(point_clouds.keys())
        )