# scCorrect
scCorrect: A robust method for integrating multi-study single-cell data

Abstract: 
The advent of single-cell sequencing technologies has revolutionized cell biology studies. However, integrative
analyses of diverse single-cell data face serious challenges, including technological noise, sample heterogeneity, and 
different modalities and species. To address these problems, we propose scCorrect, a Variational Autoencoder (VAE)-based 
model that can integrate single-cell data from different studies and map them into a common space. Specifically, we 
designed a Study Specific Adaptive Normalization (SSAN) for each study in decoder to implement these features. scCorrect 
substantially achieves competitive and robust performance compared with state-of-the-art (SOTA) methods and brings novel 
insights under various circumstances (e.g., various batches, multi-omics, cross-species, and development stages). In 
addition, the integration of single-cell data and spatial data makes it possible to transfer information between 
different studies, which greatly expands the narrow range of genes covered by MERFISH technology. In summary, scCorrect 
can efficiently integrate multi-study single-cell datasets, thereby providing broad opportunities to tackle challenges 
emerging from noisy resources.

![(Variational) gcn](Figure1.jpg)


## Getting started
### data preparation
The data can be downloaded from the Table S1 in paper. We also provide a toy cross-speices embryo development data in 
datafolder. The one-to-one homologous gene table of human and mouse is also in data folder.


### a simple tutorial

0, RNA_pancreas_real.ipynb

1, RNA_pancreas_simulate.ipynb

2, Embryo_development.ipynb

## Contact

guozhenhao17@mails.ucas.ac.cn

guozhenhao@tongji.edu.cn


## Citation

Guo, Z. H., Wang, Y. B., Wang, S., Zhang, Q., & Huang, D. S. (2024). scCorrector: a robust method for integrating 
multi-study single-cell data. Briefings in Bioinformatics, 25(2), bbad525. 
