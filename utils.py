from __future__ import print_function

import scipy.sparse as sp
import numpy as np
import networkx as nx
from sklearn import preprocessing
from keras.utils import to_categorical
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora",use_feature=True):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(path))

    idx_features_labels=np.loadtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))

    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    g=nx.read_edgelist("{}{}.cites".format(path, dataset))
    N=len(g)
    adj=nx.to_numpy_array(g,nodelist=idx_features_labels[:, 0])
    adj = sp.coo_matrix(adj)

    if use_feature:
        features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    else:
        features=np.identity(N,dtype=np.float32)
    
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], g.size(), features.shape[1]))

    return features, adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    
    y_train = np.zeros(y.shape, dtype=np.int32) #y:label
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]

    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask,val_mask,test_mask


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k
