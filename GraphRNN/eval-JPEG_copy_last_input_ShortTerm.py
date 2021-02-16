import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from models.CopyLastInput_without_color_ShortTerm import Copy_Last_Input as Model
from datasets.without_color.bodys_JPEG import Bodys as Dataset_JPEG
import tf_nndistance
import tf_approxmatch


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys', help='path')
parser.add_argument('--dataset', default='bodys', help='Dataset. argo or nu [default: argo]')
parser.add_argument('--unit', type=str, default='pointrnn', help='Unit. pointrnn, pointgru or pointlstm [default: pointrnn]')
parser.add_argument('--ckpt-step', type=int, default=200000, help='Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=4000, help='Number of points [default: 1024]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 8]')
parser.add_argument('--seq-length', type=int, default=12, help='Length of sequence [default: 10]')
parser.add_argument('--log-dir', default='outputs/Bodys_Short_Term_copy_last_input', help='Log dir [outputs/bodys-pointrnn]')
args = parser.parse_args()

args.log_dir += '/%s-%s'%(args.dataset, args.unit)

print("=====    RUN BODYS COPY LAST INPUT TESTING TESTING       ==========\n")

test_dataset = Dataset_JPEG(root=args.data_path,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train= False) # This will be false in evaluation

print("[loaded test_dataset loaded] ",test_dataset )


print("Model",Model)
model = Model(batch_size=1,
              seq_length=args.seq_length,
              num_points=args.num_points,
              num_samples=args.num_samples,
              knn=True,
              is_training=False)

checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt-%d'%args.ckpt_step)
test_dir = os.path.join(args.log_dir, 'test-examples')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

log = open(os.path.join(args.log_dir, 'test_JEPG_copy_last_input.log'), 'w')
log_frame = open(os.path.join(args.log_dir, 'test_JPEG_frames_copy_last_input.log'), 'w')


def run_test_sequence(sequence_nr, start, fps):
    print("TEST SEQUENCE: ",sequence_nr)
    
    nrs = sequence_nr, start
    batch = test_dataset[nrs]
    batch =np.array(batch)
    print("batch.shape",batch.shape)
    test_seq =batch
    test_seq =np.expand_dims(test_seq, 0)
    print("test_seq.shape: " ,test_seq.shape)
    
    feed_dict = {model.inputs: test_seq}
    
    cd, emd, downsample_frames, predictions = sess.run([model.cd, model.emd, model.downsample_frames, model.predicted_frames], feed_dict=feed_dict) 
    #Write Log
    log.write('[%s]\tTEST:[%10d:]\tSTART:%d fps:1/%d [CD,EMD]:\t%.12f\t%.12f\n'%(str(datetime.now()), sequence_nr, start, fps, cd, emd))
    
    log.flush()
    print('[%s]\tTEST:[%10d:]\tSTART:%d fps:1/%d [CD,EMD]:\t%.12f\t%.12f\n'%(str(datetime.now()), sequence_nr, start, fps, cd, emd))

    


with tf.Session() as sess:

    #Print Model Parameters
    #flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    #parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    #print ('total flops: {}'.format(flops.total_float_ops))
    #print ('total parameters: {}'.format(parameters.total_parameters))
    
    
    
    nr_of_test = 10
    for sequence_nr in range(0,nr_of_test,3):
    	
    	fps = 1
    	test_seq = run_test_sequence(sequence_nr, 2, fps)
    	test_seq = run_test_sequence(sequence_nr, 15, fps)
    	test_seq = run_test_sequence(sequence_nr, 25, fps)
    	

    nr_of_test = 11
    for sequence_nr in range(1,nr_of_test,3):
    	
    	fps = 2
    	test_seq = run_test_sequence(sequence_nr, 0, fps)
    	test_seq = run_test_sequence(sequence_nr, 12, fps)


    nr_of_test = 12
    for sequence_nr in range(2,nr_of_test,3):
    	
    	fps = 3
    	test_seq = run_test_sequence(sequence_nr, 0, fps)
  
