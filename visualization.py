import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ROOT_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
import tf_util
import pc_util

import scannet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='sem_seg', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')

parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')

FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu



MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(os.path.join(ROOT_DIR, 'models'), FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')



HOSTNAME = socket.gethostname()

NUM_CLASSES = 21

# Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR,'data')

TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test')

color_map = [[0, 0, 0],[174, 199, 232],[152, 223, 138],[31, 119, 180],[255, 187, 120],
             [188, 189, 34],[140, 86, 75],[255, 152, 150],[214, 39, 40],[197, 176, 213],
             [148, 103, 189],[196, 156, 148],[23, 190, 207],[247, 182, 210],[219, 219, 141],
             [255, 127, 14],[158, 218, 229],[44, 160, 44],[112, 128, 144],[227, 119, 194],[82, 84, 163]]

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'a+') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], 1.0*color[0]/255, 1.0*color[1]/255, 1.0*color[2]/255))

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'a+') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [9,200,248]
            elif seg[i] == 0:
                color = [1,0,0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)



def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state("log")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("成功导入模型")

        #sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'end_points': end_points}

        eval_whole_scene_one_epoch(sess, ops)

# evaluate on whole scenes to generate numbers provided in the paper
def eval_whole_scene_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    
    num_batches = len(TEST_DATASET_WHOLE_SCENE)

    
    for batch_idx in range(num_batches):
        
        batch_data, batch_label, batch_smpw = TEST_DATASET_WHOLE_SCENE[batch_idx]
        if not os.path.exists("./result"): os.mkdir("./result")
        result_right_path = "./result/"+str(batch_idx)+"_right.obj"
        result_pre_path = "./result/"+str(batch_idx)+"_pre.obj"
        result_diff_path = "./result/"+str(batch_idx)+"_diff.obj"
        for i in range(batch_data.shape[0]):
            feed_dict = {ops['pointclouds_pl']: batch_data[i:i+1,...],
                        ops['labels_pl']: batch_label[i:i+1,...],
                        ops['smpws_pl']: batch_smpw[i:i+1,...],
                        ops['is_training_pl']: is_training}
            
            pred_val = sess.run([ops['pred']], feed_dict=feed_dict)
            
            pred_val=np.squeeze(pred_val)
            pred_val = np.argmax(pred_val,axis=1) # BxN
            output_color_point_cloud(batch_data[i,...],pred_val,result_pre_path)
            output_color_point_cloud(batch_data[i,...],batch_label[i,...],result_right_path)
            output_color_point_cloud_red_blue(batch_data[i,...],pred_val==batch_label[i,...],result_diff_path)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
