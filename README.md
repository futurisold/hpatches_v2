# HPatches v2

The current format of the [HPatches](https://hpatches.github.io/) dataset may prove to be inadequate for evaluating deep learning models that typically process images, rather than patches. In order to effectively evaluate such models, it would be beneficial to have a dataset that comprises images with ground-truth keypoints, allowing for the comparison of descriptors extracted from corresponding regions (i.e. patches) in each image. The presence of a sufficient number of matches between these patches would then serve as an indicator of the robustness of the algorithm being evaluated.

As a contribution to the research community, I am pursuing the development of such a dataset. I will update this repository progressively.

# Citation
```bibtex
@InProceedings{hpatches_2017_cvpr,
author={Vassileios Balntas and Karel Lenc and Andrea Vedaldi and Krystian Mikolajczyk},
title = {HPatches: A benchmark and evaluation of handcrafted and learned local descriptors},
booktitle = {CVPR},
year = {2017}}
```
