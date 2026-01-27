# DISPATCH-DISTILLING-SELECTIVE-PATCHES-FOR-SPEECH-ENHANCEMENT
[![PWC](https://img.shields.io/badge/ICASSP-paper-red)](https://www.arxiv.org/abs/2509.15922) 

Official implementation of **"[DISPATCH: Distilling Selective Patches for Speech Enhancement](https://www.arxiv.org/abs/2509.15922)" (ICASSP 2026)**.

*In this work, we propose Distilling Selective Patches (DISPatch), a knowledge distillation (KD) framework for speech enhancement. Conventional KD methods often require a compact student model to imitate a high-capacity teacher's output entirely, which can propagate the teacher's errors and yield minimal gains in regions where the student already performs well. To address this, DISPatch selectively applies the distillation loss only to spectrogram patches where the teacher outperforms the student, as measured by a Knowledge Gap Score. This strategy focuses optimization on regions with the most significant potential for improvement, while minimizing influence from regions where the teacher may be unreliable. Furthermore, we introduce Multi-Scale Selective Patches (MSSP), an extension that uses different patch sizes across low- and high-frequency bands to account for spectral heterogeneity. Our experiments show that integrating DISPatch and MSSP into state-of-the-art KD methods consistently and considerably improves the performance of the student model.*


## Overview of DISPatch framework
![DISPATCH figure](fig/figure_1.png?v=2)

## Results
![results](fig/results.png?v=2)

## Citations
```
@inproceedings{kim2026dispatchdistillingselectivepatches,
  author    = {Dohwan Kim and Jung-Woo Choi},
  title     = {DISPATCH: Distilling Selective Patches for Speech Enhancement},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year      = {2026},
  address   = {Barcelona, Spain},
  month     = {May},
  publisher = {IEEE}
}
```
