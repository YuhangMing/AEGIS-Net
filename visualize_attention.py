#
#
#      0=========================================0
#      |    Semantic Indoor Place Recognition    |
#      0=========================================0
#
#      Yuhang Ming
#

# Common libs
import os
# import sys
import signal
import argparse
import numpy as np
import torch

# Dataset
from torch.utils.data import DataLoader
# from datasets.ScannetTriple import *
from datasets.Visualization import *

from models.architectures import KPFCNN
from models.PRNet import PRNet
from models.TransPRNet import TransPRNet
from utils.config import Config

# Visualisation
import open3d as o3d
# import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#
#
#   use single GPU, use 
#   export CUDA_VISIBLE_DEVICES=3
#   in terminal
#


if __name__ == '__main__':

    #####################
    # PARSE CMD-LINE ARGS
    #####################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_feat', type=int, default=3, help='How many block features to use [default: 5]')
    parser.add_argument('--no_att', dest='bNoAtt', action='store_true', help='Set not to use attention layers')
    parser.add_argument('--no_color', dest='bNoColor', action='store_true', help='Set not to use color in input point clouds')
    parser.add_argument('--evaluate', dest='bEVAL', action='store_true', help='Set to evaluate the VLAD results')
    parser.add_argument('--visualise', dest='bVISUAL', action='store_true', help='Set to visualise the VLAD results')
    FLAGS=parser.parse_args()
    print('\n----------------------------------------')
    print('Visualising parameters loaded from files.')
    print('Number of features:', FLAGS.num_feat)
    print('Use attention layers:', not FLAGS.bNoAtt)
    print('Use color information:', not FLAGS.bNoColor)
    print('Evaluation:', FLAGS.bEVAL)
    print('Visualisation:', FLAGS.bVISUAL)


    ######################
    # LOAD THE PRE-TRAINED 
    # SEGMENTATION NETWORK
    ######################

    print('\n----------------------------------------')
    t = time.time()
    if FLAGS.bNoColor:
        print('ScanNetSLAM, WITHOUT color')
        chosen_log = 'results/Log_2021-06-16_02-31-04'  # => ScanNetSLAM (full), w/o color, batch 8, 1st feat 64, 0.04-2.0
    else:
        print('ScanNetSLAM, WITH color')
        chosen_log = 'results/Log_2021-06-16_02-42-30'  # => ScanNetSLAM (full), with color, batch 8, 1st feat 64, 0.04-2.0
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = 0 # chkp_500
    # print('Loading pre-trained segmentation KP-FCNN from', chosen_log, 'chkp_idx=', chkp_idx)

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
    print('Found checkpoint(s):', chkps)
    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    print('Loading Checkpoint:', chosen_chkp)
    # Initialise and Load the segmentation network configs
    config = Config()
    config.load(chosen_log) # update config file
    config.KPlog = chosen_chkp
    # Change parameters for the TESTing here. 
    # For example, you can stop augmenting the input data.
    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    config.batch_num = 1        # for cloud segmentation
    config.val_batch_num = 1    # for SLAM segmentation
    #config.in_radius = 4
    config.validation_size = 50    # decide how many points will be covered in prediction -> how many forward passes
    # 50 is a suitable value to cover a room-scale point cloud
    # 4 is a suitable value to cover a rgbd slam input size point cloud
    config.input_threads = 0
    # config.print_current()

    # set label manually here for scannet segmentation
    # with the purpose of putting loading parts together
    # ScanNet SLAM
    label_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    ignored_labels = [0]
    # Initialise segmentation network
    seg_net = KPFCNN(config, label_values, ignored_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")        
    seg_net.to(device)
    # Load pretrained weights
    checkpoint = torch.load(config.KPlog)
    # print(checkpoint.keys())    # ['epoch', 'model_state_dict', 'optimizer_state_dict', 'saving_path']
    # print(checkpoint['model_state_dict'].keys())    # where weights are stored
    # print(checkpoint['optimizer_state_dict'].keys())
    seg_net.load_state_dict(checkpoint['model_state_dict'])
    # number of epoch trained
    epoch = checkpoint['epoch']
    # set to evaluation mode
    # Dropout and BatchNorm (and maybe some custom modules) behave differently during training and 
    # evaluation. You must let the model know when to switch to eval mode by calling .eval() on 
    # the model. This sets self.training to False for every module in the model. 
    seg_net.eval()
    print("SEGMENTATION model and training states restored with", epoch+1, "epoches trained.")
    print('Done in {:.1f}s'.format(time.time() - t))


    #####################
    # LOAD TRAINED 
    # RECOGNITION NETWORK
    #####################

    print('\n----------------------------------------')
    print('Load pre-trained recognition VLAD')
    print('*********************************')
    t = time.time()
    # path to the trained model
    ## ACGiS-Net Logs
    # chosen_log = 'results/Recog_Log_2023-07-13_14-20-52'    # full model trained for 30 epochs
    # chosen_log = 'results/Recog_Log_2023-07-23_08-07-54'    # full model trained for 58 epochs
    ## CGiS-Net Logs
    # chosen_log = 'results/Recog_Log_2023-08-01_09-01-30'    # no attention trained for 30 epochs
    # chosen_log = 'results/Recog_Log_2021-08-29_13-46-24'    # default CGiS-Net with 5 feats
    ## Visualization Test
    chosen_log = 'results/Recog_Log_VisTest'                # no attention trained for 30 epochs

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = 3        # -1 for latest, None for current
    print('Chosen log:', chosen_log, 'chkp_idx=', chkp_idx)
    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
    print('Checkpoints found:', np.sort(chkps))
    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    print('Checkpoints chosen:', chosen_chkp)
    # Initialise and Load the configs
    config = Config()
    config.load(chosen_log)
    # # Change parameters for the TESTing here. 
    config.validation_size = 3700    # decide how many points will be covered in prediction -> how many forward passes
    config.print_current()

    # Initialise recognition network
    if config.no_attention:
        reg_net = PRNet(config)
    else:
        reg_net = TransPRNet(config)
    reg_net.to(device)
    # Load pretrained weights
    checkpoint = torch.load(chosen_chkp)
    reg_net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    reg_net.eval()
    print("RECOGNITION model and training state restored with", epoch, "epoches trained.")
    print('Done in {:.1f}s\n'.format(time.time() - t))


    print('\nData Preparation')
    print('****************')
    t = time.time()
    test_dataset = VisualizationDataset(config, 'scene0710_00_1230_sub.ply')
    test_sampler = VisualizationSampler(test_dataset)
    # Initialize the dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                            collate_fn=VisualizationCollate, num_workers=config.input_threads,
                            pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)
    print('Calibed batch limit:', test_sampler.dataset.batch_limit)
    print('Calibed neighbor limit:', test_sampler.dataset.neighborhood_limits)
    print('Done in {:.1f}s\n'.format(time.time() - t))


    print('\nStart visualisation')
    print('**********')
    t = time.time()

    for i, batch in enumerate(test_loader):
        if i > 0:
            break
        q_fmid = batch.frame_inds.cpu().detach().numpy()[0]     # list, [scene_id, frame_id]
        print(i, '-', q_fmid)

        batch.to(device)
        print('- Segmentation Layers')
        feat = seg_net.inter_encoder_features(batch)
        print('- Embedding Layers')
        vlad, attention_weights = reg_net.get_attention_weights(feat)
        print(vlad.shape)
        for w in attention_weights:
            print(torch.sum(w))

            # print(w.shape)
        # print(attention_weights[0])

        print('\nVisualisation')
        print('*************')
        retri_colors = [[0, 0.651, 0.929],   # blue
                        [0, 0.8, 0],         # green
                        [1.0, 0.4, 0.4],     # red
                        [1, 0.706, 0]        # yellow
                        ]
        for query, retrivs in test_pair:
            # get the query point cloud
            query_file = test_loader.dataset.files[query[0]][query[1]]
            # q_pose = test_loader.dataset.poses[query[0]][query[1]]
            print('processing:', query_file)
            print('query/retrivs:', query, retrivs)

            q_pcd = o3d.io.read_point_cloud(query_file)
            # q_pcd.transform(q_pose)
            # q_pcd.paint_uniform_color([1, 0.706, 0])        # yellow
            q_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

            # visualise query in the original color
            # create visualisation window
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Query & Retrieval', width=960, height=960, left=360, top=0)
            vis.add_geometry(q_pcd)
            
            for k, retri in enumerate(retrivs):
                # get k-th retrieved point cloud
                retri_file = test_loader.dataset.files[retri[0]][retri[1]]
                # r_pose = test_loader.dataset.poses[retri[0]][retri[1]]

                r_pcd = o3d.io.read_point_cloud(retri_file)
                trans = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 4.*(k+1)], [0.0, 0.0, 0.0, 1.0]]

                # r_pcd.transform(r_pose)
                r_pcd.transform(trans)
                # r_pcd.paint_uniform_color(retri_colors[k])
                r_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
                vis.add_geometry(r_pcd)

            vis.run()
            vis.destroy_window()
