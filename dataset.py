from __future__ import print_function, division
from .mira_utils.cv_io import CVIO



def get_CV_dataset(cfg,
                augmentor,
                mode='test',
                   ):
    r"""Prepare dataset for training and inference.
    """
    global sample_stride
    assert mode in [ 'test']

    sample_label_size = cfg.MODEL.OUTPUT_SIZE
    topt, wopt = ['0'], [['0']]

    if mode == 'test':
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_stride = cfg.INFERENCE.STRIDE
        iter_num = -1

    shared_kwargs = {
        "sample_volume_size": sample_volume_size,
        "sample_label_size": sample_label_size,
        "sample_stride": sample_stride,
        "augmentor": augmentor,
        "target_opt": topt,
        "weight_opt": wopt,
        "mode": mode,
        "do_2d": cfg.DATASET.DO_2D,
        "reject_size_thres": cfg.DATASET.REJECT_SAMPLING.SIZE_THRES,
        "reject_diversity": cfg.DATASET.REJECT_SAMPLING.DIVERSITY,
        "reject_p": cfg.DATASET.REJECT_SAMPLING.P,
        "data_mean": cfg.DATASET.MEAN,
        "data_std": cfg.DATASET.STD,
        "erosion_rates": cfg.MODEL.LABEL_EROSION,

    }


    if cfg.INFERENCE.CV_RAW_NAME is not None:
        vol_size = cfg.DATASET.VOLUME_SIZE[::-1]
        mip_factor = 2 ** cfg.DATASET.MIP
        vol_size = (vol_size[0] // mip_factor,
                    vol_size[1] // mip_factor,
                    vol_size[2])
        dataset = CVIO(cfg.INFERENCE.CV_RAW_NAME, cfg.INFERENCE.CV_RESOLUTION,
                                  vol_size, cfg.INFERENCE.TARGET,
                                  pad_size=cfg.DATASET.PAD_SIZE,
                                  data_scale=cfg.DATASET.DATA_SCALE,
                                  affine=cfg.INFERENCE.affine, affine_matrix_path=cfg.INFERENCE.affine_matrix,
                       **shared_kwargs)


    return dataset



