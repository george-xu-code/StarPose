# StarPose
Official code repository for the paper: StarPose: Sparkling Lightweight Human Pose Estimation with Star Operation


## Abstract

Lightweight human pose estimation has long been a research hotspot and challenge in this field. Most existing methods primarily focus on introducing high-resolution design patterns in HRNet, followed by lightweight modifications. However, the presence of multi-resolution branches introduces a throughput bottleneck in this paradigm. 
This study proposes StarPose, a single-branch, upsampling-free network structure. The network is designed with HRPVT-S as the macro architecture, and all micro block designs are optimized from a lightweight perspective. 
Furthermore, the overall architecture is restructured using the advanced lightweight design insight: the star operation, which can handle high-dimensional features while computing in a low dimensional space. 
The proposed method achieves two times faster inference speed than Lite-HRNet with almost the same model complexity on the MS COCO and MPII benchmarks, while maintaining superior accuracy, thereby setting a new state-of-the-art performance. 

<img src="/resources/starpose.png"/>

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | #Params | GFLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | ckpt |
| :----------------- | :-----------: | :------: | :-----------: | :------: | :------: | :------:| :------: | :------: | :------: | :------: |
| [StarPose-16](/configs/body_2d_keypoint/simcc/coco/starpose-16_1e-3_260e_256x192_simcc2.py)  | 256x192 | 1.3M | 0.26 |0.672 | 0.876 | 0.746 | 0.646 | 0.725 | 0.729 | [GoogleDrive](https://drive.google.com/file/d/1XrqYIqup82F7QrqxCKnHc3VmA1-J17B9/view?usp=sharing) |
| [StarPose-16](/configs/body_2d_keypoint/simcc/coco/starpose-16_1e-3_100e_384x288_simcc2.py)  | 384x288 | 1.4M | 0.59 | 0.704 | 0.883 | 0.774 | 0.675 | 0.762 | 0.756 | [GoogleDrive](https://drive.google.com/file/d/1ilalZl7yoo_UaUXXhB1gLO6KXmF18U6Y/view?usp=sharing) |
| [StarPose-18](/configs/body_2d_keypoint/simcc/coco/starpose-18_1e-3_260e_256x192_simcc2.py)  | 256x192 | 1.9M | 0.45 | 0.700 | 0.885 | 0.775 | 0.672 | 0.756 | 0.756 | [GoogleDrive](https://drive.google.com/file/d/1qiJ-cF50wUHYCg-VTnrmAOua8D7bNMQr/view?usp=sharing) |
| [StarPose-18](/configs/body_2d_keypoint/simcc/coco/starpose-18_1e-3_100e_384x288_simcc2.py)  | 384x288 | 2.0M | 1.0 | 0.729 | 0.892 | 0.798 | 0.696 | 0.791 | 0.780 | [GoogleDrive](https://drive.google.com/file/d/169QRBW-tqHbwMZZ2jlh5uFJ-do6UEbqO/view?usp=sharing) |

### Results on MPII val set

| Arch  | Input Size | #Params | GFLOPs | PCKh@0.5 | ckpt |
| :--- | :--------: | :------: | :--------: | :------: | :------: |
| [StarPose-16](/configs/body_2d_keypoint/simcc/mpii/starpose-16_2e-3_300e_mpii_256x256_simcc2.py) | 256x256 | 1.3M | 0.4 | 0.873 | [GoogleDrive](https://drive.google.com/file/d/1h5_D9nX4Pk4uOdvXtJA2lDi5rTE1sPSm/view?usp=sharing) |
| [StarPose-18](/configs/body_2d_keypoint/simcc/mpii/starpose-18_2e-3_300e_mpii_256x256_simcc2.py) | 256x256 | 1.9M | 0.6 | 0.878 | [GoogleDrive](https://drive.google.com/file/d/1Q0BZJnFR23LyurASwkC5aGwKa2_bWwuJ/view?usp=sharing) |
## Prepare datasets

It is recommended to Symlink the dataset root to `$StarPose/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. [HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-)
Download and extract them under `$StarPose/data`, and make them look like this:

```
StarPose
├── configs
├── tools
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   |-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...

```

**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
We have converted the original annotation files into json format, please download them from [mpii_annotations](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/datasets/mpii_annotations.tar).
Extract them under `$StarPose/data`, and make them look like this:

```
StarPose
├── configs
├── tools
`── data
    │── mpii
        |── annotations
        |   |── mpii_gt_val.mat
        |   |── mpii_test.json
        |   |── mpii_train.json
        |   |── mpii_trainval.json
        |   `── mpii_val.json
        `── images
            |── 000001163.jpg
            |── 000003072.jpg

```


# Acknowledgement:
This project is developed based on the [MMPOSE](https://github.com/open-mmlab/mmpose). Please follow the official documentation for environment setup.


