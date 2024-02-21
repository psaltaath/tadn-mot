# Transformer-based decision network for multiple object tracking (TADN)
[![License: GPL 3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

Check our final paper in CVIU journal [here](https://www.sciencedirect.com/science/article/abs/pii/S1077314224000389)  
or check our pre-print publication [here](https://arxiv.org/abs/2208.03571)

### Highlights
- Optimization during inference
- TADN can directly infer assignment pairs between detections and active targets in a single forward pass of the network
- TADN is integrated in a rather simple MOT framework
- Coupled with a novel training strategy for efficient end-to-end training

## Benchmark results

| Dataset     | Detections          | MOTA          | IDF1          | MT            | ML            | FP             | FN              | IDSW          | Frag          | Hz            |
|-------------|---------------|---------------|---------------|---------------|----------------|-----------------|---------------|---------------|---------------|---------------|
| [MOT17](https://motchallenge.net/data/MOT17/) | Public | 54.6 | 49.0          | 22.4 | 30.2 | 36285          | 214857 | 4869          | 7821          | 10.0 |
| [MOT17](https://motchallenge.net/data/MOT17/) | Private | 69.0 | 60.8          | 45.7 | 13.6 | 47466          | 124623 | 2955          | 4119          | - |
| [MOT20](https://motchallenge.net/data/MOT20/) | Private | 68.7 | 61.0          | 57.4 | 14.3 | 27135          | 133045 | 1707          | 2321          | - |
| [UA-DETRAC](https://detrac-db.rit.albany.edu/) | Private | 23.7 | -          | 61.2 | 8.2 | 31417          | 198714 | 198714          | -          | - |

### Notes
- TADN performance for MOT17 in Hz is achieved using a NVIDIA Geforce GTX2080Ti. Performance may vary for different hardware configurations.
- TADN is trained **exclusively** on each benchmark's provided training data.
- MOT17 and MOT20 private detections results are achieved using a community pretrained YOLO-X detector publicly available [here](https://github.com/ifzhang/ByteTrack).
- UA-DETRAC metrics are the CLEAR metrics along the detector's PR-curve. 


![TADN MOT tracking pipeline](assets/mot_tracker.png)

![TADN possible architectures](assets/branches.png)


## Installation

To install **tadn-mot** you must first install its dependencies as found in ```requirements.txt```. For ease, you can use docker.

### Docker

To use **tadn-mot** with docker, first you must build the appropriate docker image

```bash
docker build . -t tadn-mot
```

To start a TADN-MOT container in interactive mode:
```bash
docker run -it [GPU_RELATED_OPTS] [MISC OPTIONS] tadn-mot bash
```
## Usage

TADN is configured for training in either MOT17 or UA-DETRAC benchmarks. To speed up thetraining process and allow for quick experimentation many inputs can be precomputed. However, TADN can be deployed to provide real-time tracking with minimal effort.

### Prepare Data
- MOT 17: Download dataset from this [link](https://motchallenge.net/data/MOT17/) and unzip.
- UA-DETRAC: Download dataset from this [link](https://detrac-db.rit.albany.edu/) and unzip.

> **Note :** For MOT17 benchmark detections from FRCNN, SDP, and DPM are provided. For UA-DETRAC detections from CompACT, RCNN, DPM and ACF detectors are provided. To use *EB* detections download detections from [here](https://github.com/bochinski/iou-tracker#eb-detections).

> **Note 2 :** To enable validation for UA-DETRAC using the trackeval repo, you must firts run the following convenience script: 
```python
python -m tadn.scripts.detrac_generate_MOTC_gt PATH_TO_DATASET_ROOT --dset_mode "train"
python -m tadn.scripts.detrac_generate_MOTC_gt PATH_TO_DATASET_ROOT --dset_mode "test"
```

### Choose appearance features CNN encoder:
1. ImageNet pretrained ResNet-18 
    - No further actions needed
    - Mediocre performance
2. Re-id pretrained ResNet-50
    - Use these [instructions](https://github.com/phil-bergmann/tracking_wo_bnw#training-the-re-identification-model) to pretrain the model on MOT17
    - Better performance (~ 5%)

### Precompute appearance vectors:
- MOT17 & Re-id CNN features
```python
python -m tadn.scripts.precompute_appearance_vectors PATH_TO_DATASET_ROOT --dset_type mot-challenge --dset_version MOT17 --feature_extractor reid --reid_ckpt PATH_TO_REID_CHECKPOINT
```

- MOT17 & Resnet-18 CNN features
```python
python -m tadn.scripts.precompute_appearance_vectors PATH_TO_DATASET_ROOT --dset_type mot-challenge --dset_version MOT17 --feature_extractor resnet18
```

- UA-DETRAC & Re-id CNN features
```python
python -m tadn.scripts.precompute_appearance_vectors PATH_TO_DATASET_ROOT --dset_type detrac --detector EB --feature_extractor reid --reid_ckpt PATH_TO_REID_CHECKPOINT
```

- MOT17 & Resnet-18 CNN features
```python
python -m tadn.scripts.precompute_appearance_vectors PATH_TO_DATASET_ROOT --dset_type mot-challenge --dset_version MOT17 --feature_extractor resnet18
```

### Camera Motion Compensation

If you intend to use CMC in your model, you must first precompute frame to frame affine transforms for the target benchmark using the ***ECC*** method.

- MOT17
```python
python -m tadn.scripts.precompute_ecc PATH_TO_DATASET_ROOT --dset_type mot-challenge --dset_mode "train" --dset_version MOT17
python -m tadn.scripts.precompute_ecc PATH_TO_DATASET_ROOT --dset_type mot-challenge --dset_mode "val" --dset_version MOT17
```

- UA-DETRAC
```python
python -m tadn.scripts.precompute_ecc PATH_TO_DATASET_ROOT --dset_type detrac --dset_mode "train"
python -m tadn.scripts.precompute_ecc PATH_TO_DATASET_ROOT --dset_type detrac --dset_mode "test"
```

### Train a TADN model

This repository uses a JSON based configuration system for defining:
- Model architecture
- Training parameters
- Dataset configuration
- Data input pipeline
- Logger configuration
- Miscelaneous hyperparameters

> **Note :** JSON config examples can be found in the ```sample_configs``` directory



#### Train on MOT17 or UA-DETRAC:
```python
python -m tadn.online_training PATH_TO_JSON_CONFIG
```

### Inference

Inference with a pre-trained TADN module supports the MOTChallenge format for the output results.

To perform inference on the *val/test* subsets:
```python
python -m tadn.scripts.inference PATH_TO_CKPT PATH_TO_JSON_CONFIG 
```

To perform inference on the whole dataset:
```python
python -m tadn.scripts.inference PATH_TO_CKPT PATH_TO_JSON_CONFIG --inference_train
```

### Evaluation

To evaluate a pre-trained TADN model on either MOT17 or UA-DETRAC datasets:
```python
python -m tadn.scripts.validate PATH_TO_CKPT PATH_TO_JSON_CONFIG
```
> **Note :** Which dataset you evaluate is configured in the JSON config

> **Note 2:** For MOT17, test set is unavailable. You can use ```"split": "half"``` to perform evaluation on a 50/50 split of MOT17 dataset. Similarly you can train a model on that split to replicate ablation study experiments. To evaluate on the official MOT17 test set, please use the "inference" script to submit inference results to the official evaluation server.


### Validate training strategy

To estimate expected tracking performance when training TADN using the online training strategy you can run:
```python
python -m tadn.scripts.validate_LAM PATH_TO_JSON_CONFIG
```
## Pre-trained TADN models

Download pretrained models and their json configs for MOT17 and UA-DETRAC from [zenodo](https://zenodo.org/record/7193331#.Y0fIqUpBy-Y) !


# Cite us!

If you use TADN in your research or wish to refer to the baseline results published here, please use the following BibTeX entry:
 
 ```
@article{psalta2024transformer,
  title={Transformer-based assignment decision network for multiple object tracking},
  author={Psalta, Athena and Tsironis, Vasileios and Karantzalos, Konstantinos},
  journal={Computer Vision and Image Understanding},
  pages={103957},
  year={2024},
  publisher={Elsevier}
}
 ```
