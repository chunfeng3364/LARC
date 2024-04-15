# Pretrain votenet on SR3D


## Set-up

We use the [official code](https://github.com/facebookresearch/votenet) of votenet. Please follow its [installation](https://github.com/facebookresearch/votenet?tab=readme-ov-file#installation) to set up the environment. 


## Data Preparation

You will need to download scans from ScanNet and process them according to this [link](https://github.com/referit3d/referit3d/blob/eccv/referit3d/data/scannet/README.md). This should result in a `keep_all_points_with_global_scan_alignment.pkl` file. Then, you need to 
- (1) Change the original 18 object categories of votenet to 607 object categories in SR3D. You can find the original defination in `./scannet/model_util_scannet.py`.
- (2) Store the SR3D objects into the same format as ScanNet data preparation in votenet. You can find details [here](https://github.com/facebookresearch/votenet/blob/main/scannet/README.md). 


## Pretraining

After data preparation, you can follow the [training steps](https://github.com/facebookresearch/votenet?tab=readme-ov-file#train-and-test-on-scannet) to pretrain votenet on SR3D. You can find our pretrained weights [here](https://drive.google.com/file/d/1CcV20pZJKJ5HZVfM-IZgN1zHfQYPPUiK/view?usp=drive_link). 


## Get votenet predictions

Now you can get the votenet predictions by doing inference on SR3D dataset. You can use our code `get_predictions.py`. It will save object category prediction, estimated bounding box, and the according point clouds of every object as a `.pkl` file. 
