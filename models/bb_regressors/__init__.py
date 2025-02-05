from functools import partial

from models.bb_regressors.mae_regressor import MAERegressor
from models.bb_regressors.roi_attention_temporal import ROITimeTransformer
from models.bb_regressors.roi_attentionv2 import ROITransformerV2
from models.bb_regressors.roi_gcn import ROIGCN
from models.bb_regressors.roi_pooling import ROIPooling

REGRESSORS = {
    "roi_pooling": partial(ROIPooling, detector=True),
    "roi_gcn": ROIGCN,
    "simple_roi": partial(ROIPooling, detector=False),
    "roi_transformerv2": ROITransformerV2,
    "roi_time_transformer": ROITimeTransformer,
    "mae": MAERegressor,
}
