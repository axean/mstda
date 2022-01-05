import scipy.spatial as spatial
import numpy as np
import gudhi as gd
import cupy as cpy
#from itertools import combinations

def is_gabriel(face, circumcentre, kdtree):
    _, nn = kdtree.query(circumcentre, k = len(face), workers = -1)    
    return np.array_equal(np.sort(nn), np.sort(face))

def cpy_distance_matrix(X,Y):
    distance_matrix = cpy.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        distance_matrix[i,:] = cpy.linalg.norm(
            cpy.broadcast_to(X[i,:], Y.shape) - Y,
            axis = 1,
            ord = 2
        )
    return distance_matrix

# The circumradius and circumcentre of the N-circumsphere
# of an N-simplex are obtained directly from elements of the inverse
# (CM-1) of the Cayley Menger matrix (CM ) for the N-simplex.
# https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm
def cpy_circumspheres(simplices, distance_matrix, X):
    assert all([ l == len(simplices[0]) for l in map(len,simplices) ])
    
    cayley_menger = cpy.zeros((len(simplices), len(simplices[0]) + 1, len(simplices[0]) + 1))
    cayley_menger[:,1:,0] = 1
    cayley_menger[:,0,1:] = 1
    cayley_menger[:,0,0]  = 0
    
    circumradii = cpy.zeros((len(simplices)))
    circumcentres = cpy.zeros((len(simplices), X.shape[1]))
    
    for t,tau in enumerate(simplices):
        for i,idx in enumerate(tau):
            for j,jdx in enumerate(tau):
                cayley_menger[t, 1+i,1+j] = distance_matrix[idx,jdx]
        #
        cayley_menger[t] = cpy.linalg.inv(cayley_menger[t])
        circumradii[t] = cpy.sqrt(cayley_menger[t,0,0]/-2)
        circumcentres[t] = cayley_menger[t,1:,0].dot( X[tau,:] )
    return (circumradii, circumcentres)

def alpha_complex_filtration(X):
    tri             = spatial.Delaunay(X)
    kdtree          = spatial.KDTree(X)
    X               = cpy.asarray(X)
    distance_matrix = cpy.power(cpy_distance_matrix(X,X),2)
    simplex_tree    = gd.SimplexTree()
    #----------------------------------------------

    # a Simplex tree is built from the cells of a Delaunay Triangulation
    for cell in tri.simplices:
        simplex_tree.insert(cell, filtration = np.nan)
    
    # we can also insert the nodes
    for node in range(X.shape[0]):
        simplex_tree.assign_filtration([node], filtration = 0.0)
    
    st_simplices = simplex_tree.get_simplices()
    simplices = [ simplex for simplex,_ in st_simplices ]

    faces = {
        i : [s for s in simplices if len(s) == i + 1]
        for i in reversed(range(1, X.shape[1]+1))
    }
    circumspheres = {
        i : cpy_circumspheres(faces[i], distance_matrix, X)
        for i in reversed(range(1, X.shape[1]+1))
    }
    
    for i in reversed(range(1, X.shape[1]+1)): # for i : dimension → 0 do
        circumradii, _ = circumspheres[i]
        circumradii = circumradii.get()
        
        for s,sigma in enumerate(faces[i]): # for all σ of dimension i
            if np.isnan(simplex_tree.filtration(sigma)): # if filtration(σ) is NaN then
                simplex_tree.assign_filtration(sigma, circumradii[s]) # filtration(σ) = α²(σ)
                
            # propagate alpha filtration value
            for tau, tau_f in simplex_tree.get_boundaries(sigma): # for all τ face of σ do
                if len(tau) == 1:
                    break
                
                if not np.isnan(tau_f): # if filtration(τ) is not NaN then
                    # filtration(τ) = min( filtration(τ), filtration(σ) )
                    simplex_tree.assign_filtration(tau, min(tau_f, circumradii[s]))
                else:
                    _, tau_cc = circumspheres[i-1]
                    tau_cc = tau_cc.get()
                    
                    if not is_gabriel(tau, tau_cc[ faces[i-1].index(tau) ], kdtree): # if τ is not Gabriel for σ then
                        # filtration(τ) = filtration(σ)
                        simplex_tree.assign_filtration(tau, circumradii[s])
        
    return list(simplex_tree.get_filtration())