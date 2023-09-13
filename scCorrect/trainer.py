#!/usr/bin/env python
import pandas as pd
import torch
import numpy as np
import os
import scanpy as sc

from .data import load_data, preprocess
from .net.vae import VAE
from .net.utils import clear_fig, EarlyStopping
from .metrics import batch_entropy_mixing_score, silhouette_score
from .logger import create_logger


def scCorrect(
        data_list=None,
        batch_categories=None,
        batch_key='batch',
        batch_name='batch',
        join='inner',

        profile='RNA',
        preprocessed=False,
        min_features=600,
        min_cells=3,
        target_sum=None,
        n_top_features=2000,
        chunk_size=20000,

        h_dim=10,
        batch_size=64,
        lr=2e-4,
        max_iteration=30000,
        seed=124,
        gpu=0,
        early_stop=True,
        eval=False,
        impute=None,

        plot_umap=True,
        plot_keys=['batch', 'cell_type', 'leiden', 'cell_ontology_class'],
        verbose=False,
        show=True,
        plot_size=(4, 4),
        plot_dpi=300,
        plot_frames=False,

        assess=True,

        outdir='output/',
):
    """

    Parameters
    ----------
    data_list
        A path list of AnnData matrices to concatenate with. Each matrix is referred to as a 'batch'.
    batch_categories
        Categories for the batch annotation. By default, use increasing numbers.
    profile
        Specify the single-cell profile, RNA or ATAC. Default: RNA.
    join
        Use intersection ('inner') or union ('outer') of variables of different batches. 
    batch_key
        Add the batch annotation to obs using this key. By default, batch_key='batch'.
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
    lr
        Learning rate. Default: 2e-4.
    max_iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
    seed
        Random seed for torch and numpy. Default: 124.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: 'output/'.
    projection
        Use for new dataset projection. Input the folder containing the pre-trained model. If None, don't do projection. Default: None. 
    repeat
        Use with projection. If False, concatenate the reference and projection datasets for downstream analysis. If True, only use projection datasets. Default: False.
    impute
        If True, calculate the imputed gene expression and store it at adata.layers['impute']. Default: False.
    chunk_size
        Number of samples from the same batch to transform. Default: 20000.
    ignore_umap
        If True, do not perform UMAP for visualization and leiden for clustering. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    assess
        If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results. Default: False.
    
    Returns
    -------
    The output folder contains:
    adata.h5ad
        The AnnData matrice after batch effects removal. The low-dimensional representation of the data is stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    log.txt
        Records raw data information, filter conditions, model parameters etc.
    umap.pdf 
        UMAP plot for visualization.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(gpu)
    else:
        device = 'cpu'

    os.makedirs(outdir + 'checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir + 'log.txt')

    adata = preprocess(data_list,
                       batch_categories=batch_categories,
                       join=join,
                       profile=profile,
                       target_sum=target_sum,
                       n_top_features=n_top_features,
                       chunk_size=chunk_size,
                       min_features=min_features,
                       min_cells=min_cells,
                       preprocessed=preprocessed,
                       batch_name=batch_name,
                       batch_key=batch_key,
                       log=log,
                       )

    # train config
    trainloader, testloader = load_data(adata, batch_size=batch_size)
    early_stopping = EarlyStopping(patience=10, switch=early_stop, checkpoint_file=outdir + '/checkpoint/model.pt')
    x_dim, n_domain = adata.shape[1], len(adata.obs['batch'].cat.categories)

    # model config
    model = VAE(x_dim, h_dim, n_domain=n_domain)

    # train
    log.info('model\n' + model.__repr__())
    model.fit(
        adata,
        trainloader,
        testloader,
        lr=lr,
        max_iteration=max_iteration,
        device=device,
        early_stopping=early_stopping,
        verbose=verbose,
    )

    # store
    adata.obsm['X_generate'], adata.obsm['batch_id'] = model.encodeBatch(testloader, device=device, eval=eval,
                                                                         out='generate')
    adata.obsm['X_scCorrect'] = model.encodeBatch(testloader, device=device, eval=eval)  # save latent rep
    adata.obsm['X_emb'] = adata.obsm['X_scCorrect']
    if impute:
        adata.layers['impute'] = model.encodeBatch(testloader, out='impute', batch_id=impute, device=device, eval=eval)
    log.info('Output dir: {}'.format(outdir))

    model.to('cpu')
    del model

    # plot
    if plot_umap:
        # config
        sc.settings.figdir = outdir
        sc.settings.set_figure_params(facecolor='white', frameon=plot_frames, dpi=plot_dpi, dpi_save=plot_dpi,
                                      figsize=plot_size)
        cols = plot_keys

        # preprocess
        log.info('Plot umap')
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_scCorrect')
        sc.tl.umap(adata, min_dist=0.1)
        sc.tl.leiden(adata)

        # plot
        color = [c for c in cols if c in adata.obs]
        sc.pl.umap(adata, color=color, save='_scCorrect.png', wspace=0.5, ncols=4, show=show)
        [clear_fig(sc.pl.umap(adata, color=c, title='', legend_loc=None, return_fig=True)).savefig(
            outdir + f'scCorrect_{c}.jpg')
            for c in color]

    # evaluate
    if assess:
        if len(adata.obs['batch'].cat.categories) > 1:
            entropy_score = batch_entropy_mixing_score(adata.obsm['X_umap'], adata.obs['batch'])
            log.info('batch_entropy_mixing_score: {:.3f}'.format(entropy_score))
            adata.uns['batch_entropy_mixing_score'] = entropy_score

        if 'cell_type' in adata.obs:
            sil_score = silhouette_score(adata.obsm['X_umap'], adata.obs['cell_type'].cat.codes)
            log.info("silhouette_score: {:.3f}".format(sil_score))
            adata.uns['silhouette_score'] = sil_score

    adata.write(outdir + 'adata_scCorrect.h5ad', compression='gzip')

    return adata


def label_transfer(ref, query, rep='X_scCorrect', label='cell_type', n_neighbors=5):
    """
    Inputs:
    ref
        reference containing the projected representations and labels
    query
        query containing the projected representations
    rep
        keys of the embeddings in adata.obsm
    labels
        label name in adata.obs
    
    Return:
    transferred label: np.array

    Examples:
        adata_query.obs['celltype_transfer']=label_transfer(adata_ref, adata_query, rep='latent', label='celltype')

    """

    from sklearn.neighbors import KNeighborsClassifier

    X_train = ref.obsm[rep]
    y_train = ref.obs.loc[:, label].to_numpy()
    X_test = query.obsm[rep]

    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)

    return y_pred, y_prob


def gene_impute(ref, query, rep='X_scCorrect', n_neighbors=30):
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    if not isinstance(ref.raw.var, pd.DataFrame):
        print('please set ref.raw.var')
        raise NotImplementedError

    if isinstance(ref.raw.X, csr_matrix):
        ref_raw = ref.raw.X.todense()
    else:
        ref_raw = ref.raw.X
    X_train = ref.obsm[rep]
    X_test = query.obsm[rep]

    # impute
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine').fit(X_train)
    distances, indices = knn.kneighbors(X_test)
    # weight = (1 - (distances[distances < 1]) / (np.sum(distances[distances < 1]))).reshape(distances.shape)
    weight = 1 - distances / (np.sum(distances))
    weight = weight / (weight.shape[1])
    query_impute = []

    # for i in range(len(X_test)):
    #     weight_temp = weight[i]
    #     indices_temp = indices[i]
    #     expression_temp = ref_raw[indices_temp]
    #     impute_temp = np.dot(weight_temp, expression_temp).tolist()[0]
    #     query_impute.append(impute_temp)

    # # method 1 slowly
    # print('method 1 slowly')
    # [query_impute.append(np.dot(weight[i], ref_raw[indices[i]]).tolist()[0]) for i in range(len(X_test))]

    # method 2 fast
    print('method 2 fast')
    ref_raw_t = torch.tensor(ref_raw, dtype=torch.float32)  # float32 or float64
    weight_t = torch.tensor(weight.reshape(weight.shape[0], 1, -1), dtype=torch.float32)
    indices_t = torch.tensor(indices, dtype=torch.int64)
    query_impute_t = torch.bmm(weight_t, ref_raw_t[indices_t])
    query_impute = query_impute_t.squeeze().detach().cpu().numpy()

    # store
    adata_impute = sc.AnnData(X=np.array(query_impute), obs=query.obs, var=ref.raw.var, obsm=query.obsm, uns=query.uns)
    return adata_impute


def location_pred(ref, query, rep='X_scCorrect', spatial_label='spatial', type_label='cell_type', n_neighbors=30,
                  noise_ratio=1e-2):
    """
    提高预测准确率，才能预测出稀有细胞类型和空间结构
    """

    # pred location
    from sklearn.neighbors import NearestNeighbors
    if spatial_label not in ref.obsm.keys():
        print('please set rep in ref.obsm')
        raise NotImplementedError
    ref_raw = ref.obsm[spatial_label]
    # 空间坐标归一化
    for i in range(len(ref_raw.shape)):
        ref_raw[:, i] = (ref_raw[:, i] - ref_raw[:, i].min()) / (ref_raw[:, i].max() - ref_raw[:, i].min())

    X_train = ref.obsm[rep]
    X_test = query.obsm[rep]

    # pred
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
    distances, indices = knn.kneighbors(X_test)
    weight = 1 - distances / (np.sum(distances))
    weight = weight / (weight.shape[1])

    # infer
    ref_raw_t = torch.tensor(ref_raw, dtype=torch.float32)  # float32 or float64
    weight_t = torch.tensor(weight.reshape(weight.shape[0], 1, -1), dtype=torch.float32)
    indices_t = torch.tensor(indices, dtype=torch.int64)
    query_impute_t = torch.bmm(weight_t, ref_raw_t[indices_t])
    query_impute = query_impute_t.squeeze().detach().cpu().numpy()

    # 增加随机噪声
    noise = np.random.rand(query_impute.shape[0], query_impute.shape[1]) * noise_ratio

    # store
    query.obsm['spatial_pred'] = query_impute + noise
    # adata_impute = sc.AnnData(X=np.array(query_impute), obs=query.obs, var=ref.raw.var)
    return query


# def location_pred(ref, query, rep='X_scCorrect', spatial_label='spatial', type_label='cell_type', n_neighbors=30,
#                   noise_ratio=1e-2):
#     """
#     提高预测准确率，才能预测出稀有细胞类型和空间结构
#     """
#     # train classifier
#     from sklearn.neighbors import KNeighborsClassifier
#     X_train = ref.obsm[rep]
#     y_train = ref.obs.loc[:, type_label].to_numpy()
#     X_test = query.obsm[rep]
#     y_test = query.obs.loc[:, type_label].to_numpy()  # TODO check
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#
#     # pred location
#     from sklearn.neighbors import NearestNeighbors
#     if spatial_label not in ref.obsm.keys():
#         print('please set rep in ref.obsm')
#         raise NotImplementedError
#     ref_raw = ref.obsm[spatial_label]
#     # 空间坐标归一化
#     for i in range(len(ref_raw.shape)):
#         ref_raw[:, i] = (ref_raw[:, i] - ref_raw[:, i].min()) / (ref_raw[:, i].max() - ref_raw[:, i].min())
#
#     X_train = ref.obsm[rep]
#     X_test = query.obsm[rep]
#
#     # pred
#     # method1，k个最近邻居的众数标签，最近的这个标签作为锚点，输入：表达谱、ref的空间位置坐标、ref的标签（或空间聚类的伪标签）
#     # method2，用互最近邻MNN，可能会出错，最近的那个不是正确的细胞类型，且锚点会非常少
#     # knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine').fit(X_train)
#     knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
#     distances, indices = knn.kneighbors(X_test)
#
#     # 重新筛选indices和distances
#     indices_new = []
#     distances_new = []
#     Ependymal = []
#     for i, indice in enumerate(indices):
#         if y_test[i] == 'Ependymal':
#             # Ependymal.append(y_pred[i])
#             pass
#         if y_pred[i] == 'Ependymal':
#             pass
#         if y_test[i] == 'Ependymal' and y_pred[i] == 'Ependymal':
#             print('Ependymal')
#             Ependymal.append(i)
#             pass
#         condition = list(y_train[indice] == y_pred[i])  # indice: 当前样本的k个最近的ref样本，y_pred：预测的query标签
#         indice_temp = condition.index(True)
#         indices_new.append([indice[indice_temp]])
#         distances_new.append([distances[i][indice_temp]])
#
#     indices = np.array(indices_new)
#     distances = np.array(distances_new)
#     weight = 1 - distances / (np.sum(distances))
#     weight = weight / (weight.shape[1])
#
#     # infer
#     ref_raw_t = torch.tensor(ref_raw, dtype=torch.float32)  # float32 or float64
#     weight_t = torch.tensor(weight.reshape(weight.shape[0], 1, -1), dtype=torch.float32)
#     indices_t = torch.tensor(indices, dtype=torch.int64)
#     query_impute_t = torch.bmm(weight_t, ref_raw_t[indices_t])
#     query_impute = query_impute_t.squeeze().detach().cpu().numpy()
#
#     # 增加随机噪声
#     noise = np.random.rand(query_impute.shape[0], query_impute.shape[1]) * noise_ratio
#
#     # store
#     query.obsm['spatial_pred'] = query_impute + noise
#     # adata_impute = sc.AnnData(X=np.array(query_impute), obs=query.obs, var=ref.raw.var)
#     return query