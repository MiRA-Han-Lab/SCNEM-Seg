# SCNEM-Seg
This repository contains the implementation of the segmentation pipeline for SCNEM dataset

# SCNEM dataset
SCNEM is a ssEM dataset, consisting of an entire unilateral mouse SCN (suprachiasmatic nucleus) with dimensions of 384 μm × 704 μm × 273 μm. 
Raw data and multiscale reconstructions will be available upon publication.

![image](https://github.com/MiRA-Han-Lab/SCNEM-Seg/blob/main/IMG/SCNEM%20.png)



# Dependencies
The inference pipeline is developed based on [PyTC](https://github.com/zudi-lin/pytorch_connectomics) and [cloudvolume](https://github.com/seung-lab/cloud-volume).


# Large scale inference

```python
python  main_inference.py --config-file /path/config.yaml --checkpoint /path/checkpoint.pth.tar
```


# Related projects

[PyTC](https://github.com/zudi-lin/pytorch_connectomics)

[cloudvolume](https://github.com/seung-lab/cloud-volume)



