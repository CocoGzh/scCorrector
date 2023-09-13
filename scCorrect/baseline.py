import scanpy as sc
import scanpy.external as sce
import scanorama
from scalex import SCALEX

"""
输入的是处理过的adata_hvg
"""


def bbknn_integrate(adata, batch_key='batch', n_pcs=50):
    adata_int = adata.copy()
    sc.tl.pca(adata_int)
    sc.external.pp.bbknn(adata_int, batch_key=batch_key, n_pcs=n_pcs)
    return adata_int


def harmony_integrate(adata, batch_key='batch'):
    adata_int = adata.copy()
    sc.tl.pca(adata_int)
    sce.pp.harmony_integrate(adata_int, key=batch_key)
    adata_int.obsm['X_emb'] = adata_int.obsm['X_pca_harmony']  # 以scib为标准
    return adata_int


def scanorama_integrate(adata, batch_key='batch'):
    # method 1 scib
    # sce.pp.scanorama_integrate(adata, key=batch_key)

    # method 2 manual
    batches = adata.obs.loc[:, batch_key].unique()
    adata_list = [adata[adata.obs.loc[:, batch_key] == batch, :] for batch in batches]
    corrected_list = scanorama.correct_scanpy(adata_list, return_dimred=True)
    corrected = sc.concat(corrected_list, label=batch_key)
    corrected.obsm["X_emb"] = corrected.obsm["X_scanorama"]
    adata_int = corrected.copy()
    return adata_int


def scvi_integrate(adata):
    from scvi.model import SCVI
    adata_int = adata.copy()
    SCVI.setup_anndata(adata_int, layer="counts", batch_key='batch')
    # SCVI.setup_anndata(adata_int, batch_key='batch')
    vae = SCVI(
        adata_int,
        # gene_likelihood="nb",
        n_layers=2,
        n_latent=30,
    )
    # vae.train()
    vae.train(train_size=1.0, max_epochs=100, use_gpu=True)
    adata_int.obsm["X_emb"] = vae.get_latent_representation()
    return adata_int


# # method 1 preprocessed adata and split it
# def scaleX_integrate(adata, batch_key='batch', max_iteration=30000):
#     batches = adata.obs.loc[:, batch_key].unique()
#     adata_list = [adata[adata.obs.loc[:, batch_key] == batch, :] for batch in batches]
#     adata_int = SCALEX(adata_list, batches, max_iteration=max_iteration, outdir='./scaleX/', processed=True)
#     adata_int.obsm['X_emb'] = adata_int.obsm['latent']  # 以scib为标准
#     return adata_int

# method 2 raw adata list
def scaleX_integrate(adata_list, batches, max_iteration=30000, min_features=600):
    adata_int = SCALEX(adata_list, batches, max_iteration=max_iteration, outdir='./scaleX/', processed=False, min_features=min_features)
    adata_int.obsm['X_emb'] = adata_int.obsm['latent']  # 以scib为标准
    return adata_int


if __name__ == '__main__':
    print('hello world')