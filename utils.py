from functools import partial
import numpy as np

def circumspheres(simplices, distance_matrix, X):
    assert all([ l == len(simplices[0]) for l in map(len,simplices) ])
    
    cayley_menger = np.zeros((len(simplices), len(simplices[0]) + 1, len(simplices[0]) + 1))
    cayley_menger[:,1:,0] = 1
    cayley_menger[:,0,1:] = 1
    cayley_menger[:,0,0]  = 0
    
    circumradii = np.zeros((len(simplices)))
    circumcentres = np.zeros((len(simplices), X.shape[1]))
    
    for t,tau in enumerate(simplices):
        for i,idx in enumerate(tau):
            for j,jdx in enumerate(tau):
                cayley_menger[t, 1+i,1+j] = distance_matrix[idx,jdx]
        #
        cayley_menger[t] = np.linalg.inv(cayley_menger[t])
        circumradii[t] = np.sqrt(cayley_menger[t,0,0]/-2)
        circumcentres[t] = cayley_menger[t,1:,0].dot( X[tau,:] )
    return (circumradii, circumcentres)

def create_sampling_domain(energy_tuple, sigma, N = 1000):
    peak_locations = set()
    for e in energy_tuple:
        peak_locations = peak_locations.union(e["mz"])
    peak_locations = sorted(list(peak_locations))

    domain = np.zeros((N,))
    
    indices = np.asarray(list(range(N)))
    indices = np.array_split(indices, len(peak_locations))
    
    assert len(indices) == len(peak_locations)
    
    for i, chunk in enumerate(indices):
        mu = peak_locations[i]
        domain[chunk] = np.linspace(mu - sigma, mu + sigma, len(chunk))

    return domain

def create_gaussian_mixtures(energy_tuple, sigma):
    def _mixture(ms):
        mz     = ms["mz"]
        intens = ms["intens"]
        #print()
        #print(mz)
        #print(intens)
        gauss  = list()
        for (peak, intens) in zip(mz, intens):
            # When a lambda is created, it doesn't make a copy of the variables in the enclosing scope that it uses
            lambda_gaussian = lambda x, peak, intens, sigma : intens * np.exp(-.5*np.power((x - peak)/sigma,2))
            gauss.append(
                partial(
                    lambda_gaussian,
                    peak   = peak,
                    intens = intens,
                    sigma  = sigma
                )
            )
            
        return lambda x : (mix := sum([ f(x) for f in gauss ]))/np.max(mix)
    
    mixtures = list()
    for e in energy_tuple:
        mixtures.append(_mixture(e))

    return tuple(mixtures)