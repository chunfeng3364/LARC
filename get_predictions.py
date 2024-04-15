import os
import sys
import numpy as np
import argparse
import importlib
import time
import json
import pickle
import sys
sys.path.append("<path to your votenet folder>")

parser = argparse.ArgumentParser()
parser.add_argument("-version", required=True, help="version of this inference")
parser.add_argument('-dataset', default='scannet', help='Dataset: sunrgbd or scannet [default: scannet]')
parser.add_argument('-num_point', type=int, default=40000, help='Point Number [default: 40000]')
parser.add_argument("-conf_thresh", default=0.0, type=float)
parser.add_argument("-ckpt", required=True)
parser.add_argument("-save_path", required=True)
FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from utils.pc_util import random_sampling
from models.ap_helper import parse_predictions, flip_axis_to_depth
from sunrgbd.sunrgbd_utils import extract_pc_in_box3d
from referit3d.in_out.neural_net_oriented import load_scan_related_data

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

if __name__=='__main__':
    SCAN_ROOT = "/viscam/data/scannetv2/scans"
    
    # Load SR3D scans
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(
        "/viscam/data/referit3d/sr3d/keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl")
    
    # Set file paths and dataset config
    demo_dir = os.path.join(BASE_DIR, 'demo_files') 
    if FLAGS.dataset == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(demo_dir, 'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    elif FLAGS.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet.scannet_detection_dataset import DC # dataset config
        checkpoint_path = os.path.join(BASE_DIR, "log_scannet_{}/checkpoint.tar".format(FLAGS.ckpt))
        all_pc_path = [os.path.join(SCAN_ROOT, "{}/{}_vh_clean_2.ply".format(scan_name, scan_name)) for scan_name in all_scans_in_dict.keys()]
    else:
        print('Unkown dataset %s. Exiting.'%(DATASET))
        exit(-1)

    eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': FLAGS.conf_thresh, 'dataset_config': DC}

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet') # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256, input_feature_dim=1, vote_factor=1,
        sampling='seed_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')
    
    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)"%(checkpoint_path, epoch))

    # Load and preprocess input point cloud
    net.eval() # set model to eval mode (for bn and dp)
    
    # Save inference results
    res = {}
    
    # Inference
    for pc_path in all_pc_path:
        scan_name = pc_path.split("/")[5]
        point_cloud = all_scans_in_dict[scan_name].pc
        color = all_scans_in_dict[scan_name].color
        pc = preprocess_point_cloud(point_cloud)
        print('Loaded point cloud data: %s'%(pc_path))

        # Model inference
        inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
        tic = time.time()
        with torch.no_grad():
            end_points = net(inputs)
        toc = time.time()
        print('Inference time: %f'%(toc-tic))
        end_points['point_clouds'] = inputs['point_clouds']
        pred_map_cls = parse_predictions(end_points, eval_config_dict)
        print('Finished detection. %d object detected.'%(len(pred_map_cls[0])))

        res_box = []
        for i in range(len(pred_map_cls[0])):
            pred_cls, pred_box, conf = pred_map_cls[0][i]
            pred_box = flip_axis_to_depth(pred_box)
            xmax = np.max(pred_box, axis=0)[0]
            ymax = np.max(pred_box, axis=0)[1]
            zmax = np.max(pred_box, axis=0)[2]
            xmin = np.min(pred_box, axis=0)[0]
            ymin = np.min(pred_box, axis=0)[1]
            zmin = np.min(pred_box, axis=0)[2]
            cxyz_box = [(xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2, xmax-xmin, ymax-ymin, zmax-zmin, 0.0]
            pred_pc, ins = extract_pc_in_box3d(point_cloud, pred_box)
            pred_color = color[ins, :]
            pred_points = np.concatenate((pred_pc, pred_color), axis=1)
            res_box.append((pred_cls, cxyz_box, pred_points))
        res[scan_name] = res_box

    with open("<{}/{}_conf{}.pkl".format(FLAGS.save_path, FLAGS.version, FLAGS.conf_thresh), "wb") as fp:
        pickle.dump(res, file=fp)
