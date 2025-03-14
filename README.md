# Monocular Per-Object Distance Estimation with Masked Object Modeling

This repository contains the officila implementation for our paper:

**Monocular Per-Object Distance Estimation with Masked Object Modeling**  
[arXiv:2401.03191](https://arxiv.org/abs/2401.03191)

🚧 **Work in Progress** 🚧  
This project is still under development. Expect updates and improvements.

## Overview

### Dataset Preparation
Download and prepare the datasets following the instructions in `DATASET.md` (to be added).

## Running the Code

To run experiments with the best hyperparameters for each dataset, use the provided scripts:

```bash
bash nuscenes.sh  # Run on NuScenes dataset
bash motsynth.sh  # Run on MOTSynth dataset
bash kitti.sh     # Run on KITTI dataset
```

Each script is pre-configured with optimal hyperparameters for the respective dataset.

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{panariello2025monocular,
  title={Monocular Per-Object Distance Estimation with Masked Object Modeling},
  author={Panariello, Aniello and Mancusi, Gianluca and Haj Ali, Fedy and Porrello, Angelo and Calderara, Simone and Cucchiara, Rita},
  journal={Computer Vision and Image Understanding},
  year=2025
}
```

## Contact
For questions or collaborations, please open an issue or reach out via email.

