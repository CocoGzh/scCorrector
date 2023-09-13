#!/usr/bin/env python

import numpy as np
import scipy
import scib
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor


def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score
    
    Algorithm
    -----
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.
    
    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.
        
    Returns
    -------
    Batch entropy mixing score
    """

    #     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i] / P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i] / P[i]) / a
            entropy = entropy - adapt_p[i] * np.log(adapt_p[i] + 10 ** -8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
        P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
        [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))


def evaluate_all(adata_raw, adata_int, method_name, batch_key='batch', label_key='cell_ontology_class', cluster_nmi=None,
             verbose=False):
    """
    ref: https://github.com/theislab/scib-pipeline/blob/main/scripts/metrics/metrics.py
    需要将算法的embedding存放在adata.obsm['X_emb']
    """

    if method_name in ['scanorama', 'harmony', 'pyliger', 'scvi', 'scanvi', 'scalex', 'scCorrect']:
        type_ = 'embed'
        embed = 'X_emb'
    elif method_name in ['bbknn']:
        type_ = 'knn'
        embed = 'X_pca'
    elif method_name in ['seurat']:
        type_ = 'full'
        embed = 'X_pca'
    else:
        raise NotImplementedError
    result = scib.me.metrics(
        adata_raw,  # 未整合前的数据
        adata_int,  # 整合后的数据
        batch_key=batch_key,  # 批次的key
        label_key=label_key,  # 细胞类型的key
        type_=type_,  # 整合方法类型, either knn, full or embed
        embed=embed,  # 整合后embedding的key
        # cluster_key='leiden',  # 聚类标签的key
        cluster_nmi=cluster_nmi,  # 计算各个resolution下nmi保存的文件
        verbose=verbose,  # 是否显示

        # 是否计算下列评价指标
        ari_=True,  # biological
        nmi_=True,  # biological
        silhouette_=True,  # asw batch: batch, asw label: biological
        pcr_=True,  # batch
        isolated_labels_f1_=True,  # biological
        isolated_labels_asw_=True,  # biological
        graph_conn_=True,  # batch
        kBET_=False,  # batch，win下爆内存
        clisi_=False,  # biological，32位程序不能用
        ilisi_=False,  # batch，32位程序不能用

        trajectory_=False,
        cell_cycle_=False,
        hvg_score_=False,
    )
    return result  # df
