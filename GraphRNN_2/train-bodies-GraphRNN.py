"""
Script to Train a Graph-RNN model 
with the Human Bodies Dataset for Short-Term Prediction
"""
import os
#Set up your devices
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from datasets.without_color.Bodies_Random import Bodys as Bodies_Random_without_color
# Load Short-Term Model
import models.GraphRNN_ShortTerm_models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

parser = argparse.ArgumentParser()
#Define your path to the Dataset
parser.add_argument('--data-dir', default='/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys', help='Dataset directory')
parser.add_argument('--batch-size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--num-iters', type=int, default=500000, help='Iterations to run [default: 800000]')
parser.add_argument('--save-cpk', type=int, default=100, help='Iterations to save checkpoints [default: 100]')
parser.add_argument('--save-summary', type=int, default=100, help='Iterations to update summary [default: 100]')
parser.add_argument('--save-iters', type=int, default=10000, help='Iterations to save examples [default: 10000]')
# training parameters
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate [default: 1e-4]')
parser.add_argument('--activation', type= int, default=0, help=' Activation function [default: 0=None or 1=tf.nn.relu]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients[default: 5.0 or 1e10 no clip].')
# data paramenters
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 8]')
parser.add_argument('--seq-length', type=int, default=12, help='Length of sequence [default: 12]')
parser.add_argument('--num-points', type=int, default=1000, help='Number of points [default: 1000]')
#model parameters
parser.add_argument('--mode', type=str, default='advanced', help='Basic model or advanced model [default: advanced]')
parser.add_argument('--unit', type=str, default='graphrnn', help='Unit. pointrnn, pointgru or pointlstm [default: pointlstm]')
parser.add_argument('--step-length', type=float, default=0.1, help='Step length [default: 0.1]')
parser.add_argument('--alpha', type=float, default=1.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta', type=float, default= 1.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--alpha_color', type=float, default=0.0, help='Weigh on CD color loss [default: 1.0]')
parser.add_argument('--beta_color', type=float, default=0.0, help='Weigh on EMD color loss [default: 1.0]')
parser.add_argument('--log-dir', default='bodies', help='Log dir [default: outputs/mminst]')
parser.add_argument('--version', default='v1', help='Model version')
parser.add_argument('--restore-training', type= int , default = 0 , help='restore-training [default: 0=False 1= True]')

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

# SET UP DIR
summary_dir = args.log_dir
summary_dir = 'summary_dir/bodies/' + summary_dir 
summary_dir += '-bodies-%s-%s'%( args.mode, args.unit)
summary_dir += '_'+ str(args.version)

args.log_dir = 'outputs/' +args.log_dir 
args.log_dir += '-bodies-%s-%s'%(args.mode, args.unit)
args.log_dir += '_'+ str(args.version)

print("\n ==== GRAPH-RNN for BODIES DATASET  ====== \n")

#Define training dataset
train_dataset = Bodies_Random_without_color(root=args.data_dir,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train=True)
print("[Dataset Loaded] ",train_dataset )

def get_batch(dataset, batch_size):
    batch_data = []
    for i in range(batch_size):
        sample = dataset[0]
        batch_data.append(sample)
    return np.stack(batch_data, axis=0)


# Set activation function (work-around parser)
if(args.activation == 0):
	args.activation = None
if(args.activation == 1):
	args.activation = tf.nn.relu
# Set training flag (work-around parser)
if(args.restore_training == 0):
	args.restore_training = False
if(args.restore_training == 1):
	args.restore_training = True

# Call Model
model_name = args.mode.capitalize() + 'Graph' + args.unit[5:].upper() 
Model = getattr(models, model_name)
print("Call model: ", model_name)
model = Model(batch_size=args.batch_size,
              seq_length=args.seq_length,
              num_points=args.num_points,
              num_samples=args.num_samples,
              knn=True,
              alpha=args.alpha,
              beta=args.beta,
              learning_rate=args.learning_rate,
              max_gradient_norm=args.max_gradient_norm,
              is_training=True)
  
# Summary params    
tf.summary.scalar('cd', model.cd)
tf.summary.scalar('emd', model.emd)
tf.summary.scalar('loss', model.loss)

# Gradients summary if you wish to save uncomment
"""
for grad, var in model.gradients:
	tf.summary.histogram("%s_grads" % str(var.name), grad)
	grad_mean= tf.reduce_mean(tf.abs(grad))
	tf.summary.histogram("%s_weights" % str(var.name), var)
	tf.summary.scalar(var.name, grad_mean)
"""
	
summary_op = tf.summary.merge_all()

# Checkpoint Directory
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')


# Restore Checkpoint
saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours = 5)
ckpt_number = 0
if (args.restore_training == True):
	checkpoint_path_automatic = tf.train.latest_checkpoint(checkpoint_dir)
	#checkpoint_path_automatic = 'outputs/bodies1k-bodies-advanced-graphrnn_v2/checkpoints/ckpt-65709'
	#print("checkpoint_path_automatic", checkpoint_path_automatic)
	ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
	ckpt_number=ckpt_number[5:]
	ckpt_number=int(ckpt_number)


# Example Folder
example_dir = os.path.join(args.log_dir, 'examples')         
if not os.path.exists(example_dir):
    os.makedirs(example_dir)
    
#Training Log    
log = open(os.path.join(args.log_dir, 'train_ckp_' + str(ckpt_number) +'.log'), 'w')
#write input arguments
log.write("\n ========  Training Log ========  \n")
log.write(":Input Arguments\n")
for var in args.__dict__:
	log.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )
log.flush()


# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(  config = config) as sess:


    if args.restore_training == False:
    	sess.run(tf.global_variables_initializer())
    else:
    	print ("\n** RESTORE FROM CHECKPOINT ***\ncheckpoint_path_restore: ", checkpoint_path_automatic)
    	model.saver.restore(sess, checkpoint_path_automatic)    
   
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    
    for i in range(ckpt_number,args.num_iters):
        
        # LOAD DATA
        batch_data = get_batch(dataset=train_dataset, batch_size=args.batch_size)
        batch = np.array(batch_data)
        #print("batch.shape", batch.shape)
        feed_dict = {model.inputs: batch_data}
        
        # RUN MODEL
        cd, emd, step, summary, downsample_frames, predictions, _ = sess.run([model.cd, model.emd, model.global_step, summary_op, model.downsample_frames, model.predicted_frames, model.train_op], feed_dict=feed_dict)
        
        print('[CD & EMD] [%10d:]\t%.12f\t%.12f'%(i+1, cd, emd))
	
	# WRITE LOG
        log.write('[%s]\t[%10d:]\t%.12f\t%.12f\n'%(str(datetime.now()), i+1, cd, emd))
        log.flush()
        
        #RELOAD DATA SET
        if(  ((i+1) % 1000 == 0) ):
        	print("[Dataset Loaded] ",train_dataset )         
        	train_dataset = Bodies_Random_without_color(root=args.data_dir,seq_length=args.seq_length,
                        num_points=args.num_points,
                        train=True)
    		         
        # WRITE SUMMARY
        if (i+1) % args.save_summary == 0:
        	summary_writer.add_summary(summary, step)

	# SAVE CHECKPOINT
        if (i+1) % args.save_cpk == 0:
            ckpt = os.path.join(checkpoint_path, )
            saver.save(sess, checkpoint_path, global_step=model.global_step)       

        # Save EXAMPLE AS NUMPY FILE
        if ( (i+1) % args.save_iters == 0 or i==0):
                	
        	downsample_frames = np.array(downsample_frames)
        	print("downsample_frames.shape", downsample_frames.shape)
        	downsample_frames = downsample_frames[0]
        	npy_path= os.path.join(example_dir, 'gdt_seq_' +str(i) )
        	np.save(npy_path, downsample_frames)
        	

        	predictions = np.array(predictions)
        	print("predictions.shape", predictions.shape)
        	predictions = predictions[0]
        	npy_path= os.path.join(example_dir, 'pdt_seq_' +str(i) )
        	np.save(npy_path, predictions)        	     	





