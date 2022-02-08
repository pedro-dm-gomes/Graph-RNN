import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"]="7"
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.decomposition import PCA
from datasets.Bodies_eval import Bodys as Dataset_Bodies_eval
import models.GraphRNN_ShortTerm_Color_models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

parser = argparse.ArgumentParser()

# Input Arguments
parser.add_argument('--data-path', default='/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys', help='path')
parser.add_argument('--manual-ckpt',type=int, default=0, help='Manual restore ckpt default[0] or automatic[1]')
parser.add_argument('--ckpt-step', type=int, default=200000, help='Manual Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=1000, help='Number of points [default: 4000]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=12, help='Length of sequence [default: 12]')
parser.add_argument('--num-digits', type=int, default=1, help='Number of moving digits [default: 1]')
parser.add_argument('--mode', type=str, default='advanced', help='Basic model or advanced model [default: advanced]')
parser.add_argument('--unit', type=str, default='graphrnn', help='Unit. pointrnn, pointgru or pointlstm [default: pointlstm]')

parser.add_argument('--log-dir', default='bodies_color', help='Log dir [default: outputs/mminst]')
parser.add_argument('--version', default='v1', help='Model version')

print("\n EVALUATION SCRIPT \n")

args = parser.parse_args()

# SET UP DIR
summary_dir = args.log_dir
summary_dir = 'summary_dir/bodies/' + summary_dir 
summary_dir += '-bodies-%s-%s'%( args.mode, args.unit)
summary_dir += '_'+ str(args.version)

args.log_dir = 'outputs/' +args.log_dir 
args.log_dir += '-bodies-%s-%s'%(args.mode, args.unit)
args.log_dir += '_'+ str(args.version)


# Call Model
model_name = args.mode.capitalize() + 'Graph' + args.unit[5:].upper() 
print("Call model: ", model_name)
Model = getattr(models, model_name)
model = Model(1,
              num_points=args.num_points,
              num_samples=args.num_samples,
              seq_length=args.seq_length,
              knn=True,
              is_training=False)
              

# Checkpoint Directory
print("args.log_dir,", args.log_dir)
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
print("checkpoint_dir", checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')

# Restore Checkpoint
ckpt_number = 0
checkpoint_path_automatic = tf.train.latest_checkpoint(checkpoint_dir)
ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
ckpt_number=ckpt_number[5:]
ckpt_number=int(ckpt_number)
if(args.manual_ckpt == 0):
	checkpoint_path_automatic =checkpoint_path_automatic
	log = open(os.path.join(args.log_dir, 'eval_ckp_' + str(ckpt_number) +'.log'), 'w')
if(args.manual_ckpt == 1):
	checkpoint_path = os.path.join(checkpoint_dir, 'ckpt-%d'%args.ckpt_step)
	log = open(os.path.join(args.log_dir, 'eval_ckp_' + str(args.ckpt_step) +'.log'), 'w')
	checkpoint_path_automatic =checkpoint_path

# Test Example Folder
example_dir = os.path.join(args.log_dir, 'test-examples')
example_dir = os.path.join(example_dir, '%04d'%(ckpt_number))
if not os.path.exists(example_dir):
    os.makedirs(example_dir)
    
test_dir = os.path.join(example_dir, 'PCA')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

#Evaluation  Log    


#write input arguments
log.write("\n ========  Evaluation Log ========  \n")
log.write(":Input Arguments\n")
for var in args.__dict__:
	log.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )
log.flush()

#Load Test Dataset
test_dataset = Dataset_Bodies_eval(root=args.data_path,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train= False) # This will be false in evaluation

print("[Dataset Loaded] ",test_dataset )

def run_test_sequence(sequence_nr):
    
    batch = test_dataset[sequence_nr]
    batch =np.array(batch)

    batch_points, batch_color = np.split(batch, 2, axis =0)
    batch_points = np.array(batch_points)
    batch_color = np.array(batch_color)
    batch =np.concatenate((batch_points,batch_color), axis=3)
    test_seq =batch[0]
    test_seq =np.expand_dims(test_seq, 0)
    
    feed_dict = {model.inputs: test_seq}
    out = []
    
    inputs = [
    	model.predicted_frames,
    	model.downsample_frames,
    	model.predicted_motions,
    	model.loss,
    	model.emd,
    	model.cd
    	]

    # Run Session 
    out = sess.run( inputs, feed_dict=feed_dict)
    
    [predictions,
    downsample_frames,
    predicted_motions,
    loss,
    emd, 
    cd
    ] = out
    
    
    print('[%s]\tTEST:[%10d:]\t[CD,EMD]:\t%.12f\t%.12f'%(str(datetime.now()), sequence_nr, cd, emd))
    
    #Write Log
    log.write('[%s]\tTEST:[%10d:]\t[CD,EMD]:\t%.12f\t%.12f\n'%(str(datetime.now()), sequence_nr, cd, emd))
    log.flush()
    
    """
    # Save Downnsample examples 
    downsample_frames = np.array(downsample_frames)
    print("downsample_frames.shape:",downsample_frames.shape)
    if (downsample_frames.shape[0] != 1): #reshape
    	print("--reshape--")
    	downsample_frames = np.reshape(downsample_frames,(downsample_frames.shape[1], downsample_frames.shape[0], downsample_frames.shape[2],downsample_frames.shape[3]))
    downsample_frames = downsample_frames[0]
    print("downsample_frames.shape:",downsample_frames.shape)
    npy_path= os.path.join(example_dir, 'gdt_test_' +str(sequence_nr) )
    np.save(npy_path, downsample_frames)

    #Save Prediction
    pc_prediction = predictions[0]
    print("pc_prediction.shape:",pc_prediction.shape)
    npy_path= os.path.join(example_dir, 'pdt_test_' +str(sequence_nr) )
    np.save(npy_path, pc_prediction)   
    """
       
                    
    return (cd, emd)



with tf.Session() as sess:
    
    # Restore Model
    print("Restore from :",checkpoint_path_automatic)
    model.saver.restore(sess, checkpoint_path_automatic)

    #flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    #parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    #log.write ('\n flops: [%s]'% (flops))
    #log.write ('\n parameters: [%s]'% (parameters))
    #log.write ('\ntotal flops: {}'.format(flops.total_float_ops))
    #log.write ('\ntotal parameters: {}'.format(parameters.total_parameters))
    #log.flush()
    
    t_cd= t_emd =0

    for seq in range(0,152):
    	cd, emd = run_test_sequence(seq)
    	t_cd = t_cd +cd
    	t_emd = t_emd + emd
    	

    print("Total CD: ",t_cd)
    print("Total EMD: ",t_emd)
    print("Total CD/152: ",t_cd/152)
    print("Total EMD/152: ",t_emd/152)     
    log.write(" Total CD  : %f \n"%(t_cd) )
    log.write(" Total EMD : %f \n"%(t_emd) )	
    log.write(" Total CD/152 : %f \n"%(t_cd/152) )
    log.write(" Total EMD/152 : %f\n"%(t_emd/152) )

