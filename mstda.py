from gtda.homology import EuclideanCechPersistence as Persistence
from gtda.diagrams import PersistenceEntropy
from gtda.pipeline import Pipeline
from gtda.diagrams import Filtering

from mstda_utils import make_point_cloud, print_dict
from mstda_utils import split_energy_levels

import numpy as np
from functools import partial
import json
import os.path
import scipy

class Model:
    def __init__(
        self,
        json_file           : str   = None,
        energy_levels       : int   = 3,
        transform           : bool  = True,
        log_x               : bool  = True,
        diagram_epsilon     : float = 0.01,
        homology_dimensions : list  = [0,1,2],
        normalized_entropy  : bool  = False,
        n_jobs              : int   = -1,
    ):
        if json_file == None:
            return # Blank model
        
        assert type(json_file) is str, "json_file argument should be a path"
        assert os.path.exists(json_file), json_file + " missing?"
        assert json_file.lower().endswith(".json"), json_file + " is not a .json file?"
        
        with open(json_file, mode = "r") as data:
            self.db = json.load(data)

        self.energy_levels       = energy_levels
        self.transform           = transform
        self.log_x               = log_x
        self.diagram_epsilon     = diagram_epsilon
        self.homology_dimensions = homology_dimensions
        self.normalized_entropy  = normalized_entropy
        self.n_jobs              = n_jobs
        
        self.pipeline  = Pipeline([
            ('diagram', Persistence(
                homology_dimensions = homology_dimensions,
                n_jobs              = n_jobs
            )),
            ('filter', Filtering(
                epsilon             = diagram_epsilon
            )),
            ('entropy', PersistenceEntropy(
                normalize           = normalized_entropy,
                n_jobs              = n_jobs
            )),
        ])
        (self.features, self.molecules) = self._compute_features(self.db)
        self.kdtree = scipy.spatial.KDTree(self.features)
    
    def query(
        self,
        mass_specs : dict,
        K : int
    ):
        (features,_) = self._compute_features(mass_specs)
        nnq = self.kdtree.query(
            features,
            k = min(K,len(self.molecules)),
            p = 2,
            workers = self.n_jobs
        )[1]
        
        point_clouds = self._make_point_clouds(mass_specs, raw = True)
        energies = split_energy_levels(point_clouds.values(),levels = self.energy_levels)

        result = dict()
        for idx, label in enumerate(point_clouds.keys()):
            nn = nnq[idx]
            nhood = dict()
            for n in nn:
                name = self.molecules[n]
                nhood[name] = self._get_frags(name)
            
            nhood_mols = list()
            for name, frags in nhood.items():
                mols = list()
                for e in energies:
                    mz_from_query = e[idx][:,0]
                    # select fragments close to the query
                    select = [
                        np.isclose(mz_from_query, b = x, atol = 5e-5).any()
                        for x in frags[:,0].astype(float)
                    ]
                    mols.append(frags[select,:])
                mols = tuple(mols)
                mols = np.concatenate(mols, axis = 0)
                nhood_mols.append(mols)
                
            nhood_mols = tuple(nhood_mols)
            nhood_mols = np.concatenate(nhood_mols, axis = 0)
            nhood_mols = np.unique(nhood_mols, axis = 0)
            nhood_mols = nhood_mols[ nhood_mols[:,0].astype(float).argsort() ]
            result[label] = nhood_mols
            
        return result
                
    def __str__(self) -> str:
        if hasattr(self, "features"):
            return "\n".join([
                "Trained model:",
                str("Molecules:\t\t\t")                 + str(len(self.molecules)),
                str("Topological features:\t\t")        + str(self.features.shape),
                str("Homology dimensions:\t\t")         + str(self.homology_dimensions),
                str("Energy levels:\t\t\t")             + str(self.energy_levels),
                str("Mass transformed:\t\t"             + str(self.transform)),
                str("Logarithmic integer mass:\t"       + str(self.log_x)),
                str("Persistence diagram epsilon:\t"    + str(self.diagram_epsilon)),
                str("Persistence entropy normalized:\t" + str(self.normalized_entropy))
            ]).expandtabs()
        else:
            return "Blank model"
        
    @classmethod
    def load(cls, saved_model_path : str):
        assert type(saved_model_path) is str, "saved_model_path should be a path"
        assert os.path.exists(saved_model_path), saved_model_path + " missing?"
        
        model = Model()
        with open(saved_model_path, mode = "r") as j:
            save_dict = json.load(j)
        
        for s,setting in save_dict["settings"].items():
            setattr(model, s, setting)
        
        setattr(model, "features", np.asarray(save_dict["features"]))
        setattr(model, "molecules", save_dict["molecules"])
        
        model.pipeline = Pipeline([
            ('diagram', Persistence(
                homology_dimensions = model.homology_dimensions,
                n_jobs              = model.n_jobs
            )),
            ('filter', Filtering(
                epsilon             = model.diagram_epsilon
            )),
            ('entropy', PersistenceEntropy(
                normalize           = model.normalized_entropy,
                n_jobs              = model.n_jobs
            )),
        ])
        
        return model
    
    def save(self, save_model_path : str):
        save_dict = dict()
        save_dict["settings"] = dict()
        save_dict["settings"]["energy_levels"]       = self.energy_levels
        save_dict["settings"]["transform"]           = self.transform
        save_dict["settings"]["log_x"]               = self.log_x
        save_dict["settings"]["diagram_epsilon"]     = self.diagram_epsilon
        save_dict["settings"]["homology_dimensions"] = self.homology_dimensions
        save_dict["settings"]["normalized_entropy"]  = self.normalized_entropy
        save_dict["settings"]["n_jobs"]              = self.n_jobs        
        
        save_dict["features"]  = [ list(f) for f in self.features ]
        save_dict["molecules"] = self.molecules
        
        with open(save_model_path, mode = "w") as j:
            json.dump(save_dict, j)
        
        print("saved model to: " + os.path.abspath(save_model_path))
    
    def _get_frags(
        self,
        molecule
    ):
        return np.asarray(self.db[molecule]["frag"])
    
    def _make_point_clouds(
        self,
        mass_specs : dict,
        raw        : bool = False,
    ) -> dict :
        return {
            k : make_point_cloud(
                v,
                transform = self.transform if not raw else False,
                log_x     = self.log_x     if not raw else False,
                levels    = self.energy_levels
            ) for k, v in mass_specs.items()
        }
    
    def _compute_features(
        self,
        mass_specs : dict
    ) -> tuple :
        point_clouds = self._make_point_clouds(mass_specs)
        features = tuple(map(
            self.pipeline.fit_transform,
            split_energy_levels(
                point_clouds.values(),
                levels = self.energy_levels
            )
        ))
        return (
            np.concatenate(features, axis = 1),
            list(point_clouds.keys())
        )
