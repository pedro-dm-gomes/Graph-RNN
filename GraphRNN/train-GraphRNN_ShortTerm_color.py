import os
#os.environ['OPENBLAS_NUM_THREADS'] = '3'
import sys
import io
from datetime import datetime
#import open3d
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from PIL import Image
import models.GraphRNN_color_ShortTerm as models
from datasets.bodys_color_Full_Random import Bodys as Dataset_Full_Random


import gc

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys', help='Dataset directory [default: data/argo-5m]')
parser.add_argument('--dataset', default='bodys', help='Dataset. argo or nu [default: argo]')
parser.add_argument('--batch-size', type=int, default= 2,help='Batch Size during training [default: 4]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 100000]')
parser.add_argument('--save-iters', type=int, default= 50, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--save-examples', type=int, default= 1000, help='Iterations to save examples [default: 1000]')
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate [default: 1e-5]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients to this norm [default: 5.0].')
parser.add_argument('--seq-length', type=int, default= 12, help='Length of sequence [default: 12]')
parser.add_argument('--num-points', type=int, default=4000, help='Number of points [default: 4000]')
parser.add_argument('--num-samples', type=int, default= 8 , help='Number of samples [default: 8]')
parser.add_argument('--mode', type=str, default='advanced', help='Basic model or advanced model [default: advanced]')
parser.add_argument('--unit', type=str, default='graphrnn', help='Unit. graphrnn, pointgru or pointlstm [default: pointrnn]')
parser.add_argument('--alpha', type=float, default=1.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta', type=float, default=1.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--alpha_color', type=float, default=0.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta_color', type=float, default=0.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--log-dir', default='outputs/GraphRNN_ShortTerm_Color', help='Log dir [default: outputs]')

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)
args.log_dir += '/%s-%s'%(args.dataset, args.unit)

print("=====    RUN BODYS TRAINING WITH COLOR   ==========\n")

"""  # =====    LOAD DATASET   ==========  """
train_dataset = Dataset_Full_Random(root=args.data_dir,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train=True)
print("[loaded train_dataset loaded] ",train_dataset )

def get_batch(dataset, batch_size):
    batch_data = []
    for i in range(batch_size):
        sample = dataset[0]
        batch_data.append(sample)
    return np.stack(batch_data, axis=0)


"""  # =====    LOAD MODEL   ==========  """
model_name = args.mode.capitalize() + 'Graph' + args.unit[5:].upper() #Returns PointRNN/LSTM
Model = getattr(models, model_name)
print("Call model: ", model_name, "\n")
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

"""  # =====    SUMMARY   ==========  """
tf.summary.scalar('cd', model.cd)
tf.summary.scalar('emd', model.emd)
tf.summary.scalar('loss', model.loss)
summary_op = tf.summary.merge_all()


"""
# =====    CHECKPOINT   ========== 
"""
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')
checkpoint_path_restore = os.path.join(checkpoint_dir, 'ckpt-74250')

print("\n\checkpoint_dir",checkpoint_dir)
checkpoint_path_automatic = tf.train.latest_checkpoint(checkpoint_dir)
print("checkpoint_path_automatic",checkpoint_path_automatic)


example_dir = os.path.join(args.log_dir, 'examples')
if not os.path.exists(example_dir):
    os.makedirs(example_dir)

log = open(os.path.join(args.log_dir, 'train.log'), 'w')
log_extra = open(os.path.join(args.log_dir, 'extra.log'), 'w')



config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("\n=====    SESSION       ========== ")
with tf.Session( config = config) as sess:
    
    sess.run(tf.global_variables_initializer())
    
    #print ("\n** RESTORE FROM CHECKPOINT ***\ncheckpoint_path_restore: ", checkpoint_path_automatic)
    #model.saver.restore(sess, checkpoint_path_automatic)
          
    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'summary'), sess.graph)
    
    print(" ")
    for i in range(0, args.num_iters):
    
    	# Load Batch data
    	batch = get_batch(dataset=train_dataset, batch_size=args.batch_size)
    	print("batch", batch.shape)
    	batch_points, batch_color = np.split(batch, 2, axis =1)
    	batch_points = np.array(batch_points)
    	batch_color = np.array(batch_color)
    	batch =np.concatenate((batch_points,batch_color), axis=4)
    	batch = np.reshape(batch,(batch.shape[1], batch.shape[0], batch.shape[2], batch.shape[3],batch.shape[4]))
    	batch =batch[0]
    	print("batch_data ", batch.shape)
    	feed_dict = {model.inputs: batch}
    	
        
    	#Call Model
    	cd, emd, downsample_frames, step, summary, predictions, _ = sess.run([model.cd, model.emd, model.downsample_frames, model.global_step, summary_op,model.predicted_frames, model.train_op], feed_dict=feed_dict)
    	
    	
    	#Write Log
    	log.write('[%s]\t[%10d:]\t%.12f\t%.12f\n'%(str(datetime.now()), i+1, cd, emd))
    	log.flush()
    	summary_writer.add_summary(summary, step)
    	print('[GEO] [%10d:]\t%.12f\t%.12f \n'%(i+1, cd, emd))


    	# Save Checkpoint
    	if ( (i+1) % args.save_iters == 0  ):
    		ckpt = os.path.join(checkpoint_path, )
    		model.saver.save(sess, checkpoint_path, global_step=model.global_step)
 		  	
	#RELOAD DATA SET
    	if(  ((i+1) % 50 == 0) ):
    		train_dataset = Dataset_Full_Random(root=args.data_dir,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train=True)
    		print("[loaded train_dataset loaded] ",train_dataset )                
    		
    	# Save examples
    	if ( ((i+1) % args.save_examples  == 0) or (i == 0) ):
    	
    		
    		#Save Batch File (better ground thruth)
    		print("Save batch file")
    		print("batch.shape",batch.shape)
    		save_batch = batch[0]
    		npy_path= os.path.join(example_dir, 'batch_' + str(i+1) )
    		np.save(npy_path, save_batch)
    		
		#Save Downsample Ground truth
    		#print("Save example")		
    		#downsample_frames = np.array(downsample_frames)
    		#print("downsample_frames",downsample_frames.shape)
    		#downsample_frames = np.reshape(downsample_frames,(downsample_frames.shape[1], downsample_frames.shape[0], downsample_frames.shape[2],downsample_frames.shape[3]))
    		#Only save the frist of the bacth size
    		#downsample_frames = downsample_frames[0]
    		#print("downsample_frames.shape:",downsample_frames.shape)
    		#npy_path= os.path.join(example_dir, 'gdt_' + str(i+1) )
    		#np.save(npy_path, downsample_frames)
    		
    		#Save Prediction
    		#Only save the frist of the bacth size
    		pc_prediction = predictions[0]
    		npy_path= os.path.join(example_dir, 'pdt_' +str(i+1) )
    		np.save(npy_path, pc_prediction)
    		print("\n")		
    
    
 


 
