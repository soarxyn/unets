# Datasets 📊

This project is configured to use the **Cityscapes** dataset.

1. Download the dataset from the official [Cityscapes](https://www.cityscapes-dataset.com/) website. You will need the `leftImg8bit` and `gtFine` packages.

2. Unzip the files and organize them into the following directory structure:

```txt
data/
├── gtFine/
│   ├── train/
│   ├── val/
│   └── test/
└── leftImg8bit/
    ├── train/
    ├── val/
    └── test/
```

3. Update the `data_dir` path in the training scripts to point to this `data/` directory.

## Citations
```
@inproceedings{Cordts2016Cityscapes,
    title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
    author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
    booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016}
}
```
