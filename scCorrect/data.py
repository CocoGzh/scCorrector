#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse, csr

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

from glob import glob

np.warnings.filterwarnings('ignore')
CHUNK_SIZE = 20000


def concat_data(
        data_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        index_unique=None,
        save=None
):
    """
    Concatenate multiple datasets along the observations axis with name ``batch_key``.
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a “batch”.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_key
        Add the batch annotation to obs using this key. Default: 'batch'.
    index_unique
        Make the index unique by joining the existing index names with the batch category, using index_unique='-', for instance. Provide None to keep existing indices.
    save
        Path to save the new merged AnnData. Default: None.
        
    Returns
    -------
    New merged AnnData.
    """
    if len(data_list) == 1:
        raise NotImplementedError
    elif isinstance(data_list, AnnData):
        return data_list

    adata_list = []
    for root in data_list:
        if isinstance(root, AnnData):
            adata = root
        else:
            raise NotImplementedError
        adata_list.append(adata)

    if batch_categories is None:
        batch_categories = list(map(str, range(len(adata_list))))
    else:
        assert len(adata_list) == len(batch_categories)
    # [print(b, adata.shape) for adata,b in zip(adata_list, batch_categories)]
    # concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key,
    #                              batch_categories=batch_categories, index_unique=index_unique)
    concat = sc.concat([*adata_list], join='inner', label=batch_key, keys=batch_categories)
    if save:
        concat.write(save, compression='gzip')
    return concat


def preprocessing_rna(
        adata: AnnData,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,  # or gene list
        batch_key='batch',
        chunk_size: int = CHUNK_SIZE,
        log=None
):
    """
    Preprocessing single-cell RNA-seq data
    
    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs × n_vars. Rows correspond to cells and columns to genes.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 600.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    log
        If log, record each operation in the log file. Default: None.
        
    Return
    -------
    The AnnData object after preprocessing.
    """
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 2000
    if target_sum is None: target_sum = 10000

    if log: log.info('Preprocessing')
    # if not issparse(adata.X):
    if type(adata.X) != csr.csr_matrix:
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata = adata[:, [gene for gene in adata.var_names
                      if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    # counts
    adata.layers['counts'] = adata.X

    if log: log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_features)

    if log: log.info('Filtering features')
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if log: log.info('Normalizing total per cell')
    sc.pp.normalize_total(adata, target_sum=target_sum)

    if log: log.info('Log1p transforming')
    sc.pp.log1p(adata)

    adata.raw = adata
    if log: log.info('Finding variable features')
    if type(n_top_features) == int and n_top_features > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key)
        adata = adata[:, adata.var.highly_variable]
    elif type(n_top_features) != int:
        adata = reindex(adata, n_top_features)

    # if log: log.info('Batch specific maxabs scaling')
    # adata = batch_scale(adata, chunk_size=chunk_size)

    if log: log.info('Processed dataset shape: {}'.format(adata.shape))
    return adata


def preprocessing_atac(
        adata: AnnData,
        min_features: int = 100,
        min_cells: int = 3,
        target_sum=None,
        n_top_features=100000,  # or gene list
        chunk_size: int = CHUNK_SIZE,
        log=None
):
    """
    Preprocessing single-cell ATAC-seq

    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs × n_vars. Rows correspond to cells and columns to genes.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 100.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization. Default: None.
    n_top_features
        Number of highly-variable features to keep. Default: 30000.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    log
        If log, record each operation in the log file. Default: None.

    Return
    -------
    The AnnData object after preprocessing.
    """
    import episcanpy as epi

    if min_features is None: min_features = 100
    if n_top_features is None: n_top_features = 10000
    if target_sum is None: target_sum = 10000

    if log: log.info('Preprocessing')
    # if not issparse(adata.X):
    if type(adata.X) != csr.csr_matrix:
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata.X[adata.X > 1] = 1

    if log: log.info('Filtering cells')
    sc.pp.filter_cells(adata, min_genes=min_features)

    if log: log.info('Filtering features')
    sc.pp.filter_genes(adata, min_cells=min_cells)

    #     adata.raw = adata
    if log: log.info('Finding variable features')
    if type(n_top_features) == int and n_top_features > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key='batch', inplace=False, subset=True)
        # epi.pp.select_var_feature(adata, nb_features=n_top_features, show=False, copy=False)
    elif type(n_top_features) != int:
        adata = reindex(adata, n_top_features)

    # if log: log.info('Normalizing total per cell')
    # if target_sum != -1:
    # sc.pp.normalize_total(adata, target_sum=target_sum)

    if log: log.info('Batch specific maxabs scaling')
    adata = batch_scale(adata, chunk_size=chunk_size)
    if log: log.info('Processed dataset shape: {}'.format(adata.shape))
    return adata


def preprocessing(
        adata: AnnData,
        profile: str = 'RNA',
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = None,
        n_top_features=None,  # or gene list
        batch_key='batch',
        chunk_size: int = CHUNK_SIZE,
        log=None
):
    """
    Preprocessing single-cell data
    
    Parameters
    ----------
    adata
        An AnnData matrice of shape n_obs × n_vars. Rows correspond to cells and columns to genes.
    profile
        Specify the single-cell profile type, RNA or ATAC, Default: RNA.
    min_features
        Filtered out cells that are detected in less than n genes. Default: 100.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
    target_sum
        After normalization, each cell has a total count equal to target_sum. If None, total count of each cell equal to the median of total counts for cells before normalization.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    log
        If log, record each operation in the log file. Default: None.
        
    Return
    -------
    The AnnData object after preprocessing.
    
    """
    if profile == 'RNA':
        return preprocessing_rna(
            adata,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            n_top_features=n_top_features,
            batch_key=batch_key,
            chunk_size=chunk_size,
            log=log
        )
    elif profile == 'ATAC':
        return preprocessing_atac(
            adata,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            n_top_features=n_top_features,
            chunk_size=chunk_size,
            log=log
        )
    else:
        raise ValueError("Not support profile: `{}` yet".format(profile))


def batch_scale(adata, chunk_size=CHUNK_SIZE):
    """
    Batch-specific scale data
    
    Parameters
    ----------
    adata
        AnnData
    chunk_size
        chunk large data into small chunks
    
    Return
    ------
    AnnData
    """
    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch'] == b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        for i in range(len(idx) // chunk_size + 1):
            adata.X[idx[i * chunk_size:(i + 1) * chunk_size]] = scaler.transform(
                adata.X[idx[i * chunk_size:(i + 1) * chunk_size]])

    return adata


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list
    
    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks
        
    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.csr_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0] // chunk_size + 1):
            new_X[i * chunk_size:(i + 1) * chunk_size, idx] = adata[i * chunk_size:(i + 1) * chunk_size, genes[idx]].X
        adata = AnnData(new_X, obs=adata.obs, var={'var_names': genes})
    return adata


class BatchSampler(Sampler):
    """
    Batch-specific Sampler
    sampled data of each batch is from the same dataset.
    """

    def __init__(self, batch_size, batch_id, drop_last=False):
        """
        create a BatchSampler object
        
        Parameters
        ----------
        batch_size
            batch size for each sampling
        batch_id
            batch id of all samples
        drop_last
            drop the last samples that not up to one batch
            
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_id = batch_id

    def __iter__(self):
        batch = {}
        sampler = np.random.permutation(len(self.batch_id))
        for idx in sampler:
            c = self.batch_id[idx]
            if c not in batch:
                batch[c] = []
            batch[c].append(idx)

            if len(batch[c]) == self.batch_size:
                yield batch[c]
                batch[c] = []

        for c in batch.keys():
            if len(batch[c]) > 0 and not self.drop_last:
                yield batch[c]

    def __len__(self):
        if self.drop_last:
            return len(self.batch_id) // self.batch_size
        else:
            return (len(self.batch_id) + self.batch_size - 1) // self.batch_size


class SingleCellDataset(Dataset):
    """
    Dataloader of single-cell data
    """

    def __init__(self, adata):
        """
        create a SingleCellDataset object
            
        Parameters
        ----------
        adata
            AnnData object wrapping the single-cell data matrix
        """
        self.adata = adata
        self.shape = adata.shape

    def __len__(self):
        return self.adata.X.shape[0]

    def __getitem__(self, idx):
        x = self.adata.X[idx].toarray().squeeze()
        domain_id = self.adata.obs['batch'].cat.codes[idx]
        return x, domain_id, idx


def preprocess(data_list,
               batch_categories=None,
               profile='RNA',
               join='inner',
               batch_key='batch',
               batch_name='batch',
               min_features=600,
               min_cells=3,
               target_sum=None,
               n_top_features=None,
               preprocessed=True,
               MinMaxScale=True,
               chunk_size=CHUNK_SIZE,
               log=None,
               ):
    # adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
    adata = sc.concat([*data_list], join=join, label=batch_key, keys=batch_categories)
    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    if batch_name != 'batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    if isinstance(n_top_features, str):
        if os.path.isfile(n_top_features):
            n_top_features = np.loadtxt(n_top_features, dtype=str)
        else:
            n_top_features = int(n_top_features)

    if preprocessed:
        adata = preprocessing(
            adata,
            profile=profile,
            min_features=min_features,
            min_cells=min_cells,
            target_sum=target_sum,
            n_top_features=n_top_features,
            chunk_size=chunk_size,
            log=log,
            batch_key=batch_key,
        )
    else:
        adata.raw = adata
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key)
        adata = adata[:, adata.var.highly_variable]
    # scale
    if MinMaxScale:
        if log: log.info('Batch specific maxabs scaling')
        adata = batch_scale(adata, chunk_size=chunk_size)
    return adata


def load_data(
        adata,
        batch_size=64,
):
    """
    Load dataset with preprocessing
    
    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. Default: 'inner'.
    batch_key
        Add the batch annotation to obs using this key. Default: 'batch'.
    batch_name
        Use this annotation in obs as batches for training model. Default: 'batch'.
    min_features
        Filtered out cells that are detected in less than min_features. Default: 600.
    min_cells
        Filtered out genes that are detected in less than min_cells. Default: 3.
    n_top_features
        Number of highly-variable genes to keep. Default: 2000.
    batch_size
        Number of samples per batch to load. Default: 64.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    log
        If log, record each operation in the log file. Default: None.
    
    Returns
    -------
    adata
        The AnnData object after combination and preprocessing.
    trainloader
        An iterable over the given dataset for training.
    testloader
        An iterable over the given dataset for testing
    """
    # adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
    # if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    # if batch_name != 'batch':
    #     adata.obs['batch'] = adata.obs[batch_name]
    # if 'batch' not in adata.obs:
    #     adata.obs['batch'] = 'batch'
    # adata.obs['batch'] = adata.obs['batch'].astype('category')
    #
    # if isinstance(n_top_features, str):
    #     if os.path.isfile(n_top_features):
    #         n_top_features = np.loadtxt(n_top_features, dtype=str)
    #     else:
    #         n_top_features = int(n_top_features)
    #
    # if preprocessed:
    #     adata = preprocessing(
    #         adata,
    #         profile=profile,
    #         min_features=min_features,
    #         min_cells=min_cells,
    #         target_sum=target_sum,
    #         n_top_features=n_top_features,
    #         chunk_size=chunk_size,
    #         log=log,
    #     )
    #
    # # scale
    # if log: log.info('Batch specific maxabs scaling')
    # adata = batch_scale(adata, chunk_size=chunk_size)

    # dataset
    scdata = SingleCellDataset(adata)
    trainloader = DataLoader(
        scdata,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    batch_sampler = BatchSampler(batch_size, adata.obs['batch'], drop_last=False)
    testloader = DataLoader(scdata, batch_sampler=batch_sampler)

    return trainloader, testloader
