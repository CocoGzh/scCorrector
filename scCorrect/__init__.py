# from pkg_resources import get_distribution

# __version__ = get_distribution('scCorrect').version
__version__ = '0.0.1'
__author__ = 'gzh'
__email__ = 'guozhenhao@tongji.edu.cn'

from .trainer import scCorrect, label_transfer, gene_impute, location_pred
