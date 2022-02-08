"""
Script to Train a Graph-RNN model 
with the MNIST Dataset for Long-Term Prediction

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
from datasets.mmnist import MMNIST

# Load Long-Term Model
import models.GraphRNN_LongTerm_models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


parser = argparse.ArgumentParser()
#Define your path to the Dataset
parser.add_argument('--data-dir', default='/home/uceepdg/profile.V6/Desktop/MNIST/', help='Dataset directory [default: data/mnist]')
parser.add_argument('--batch-size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--num-iters', type=int, default=500000, help='Iterations to run [default: 500000]')
parser.add_argument('--save-cpk', type=int, default=10, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--save-summary', type=int, default=100, help='Iterations to update summary [default: 20]')
parser.add_argument('--save-iters', type=int, default=10000, help='Iterations to save examples [default: 100000]')
# training parameters
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate [default: 1e-4]')
parser.add_argument('--activation', type= int, default=0, help=' Activation function [default: 0=None or 1=tf.nn.relu]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients[default: 5.0 or 1e10 = no clip].')
# data paramenters
parser.add_argument('--num-points', type=int, default=128, help='Number of points [default: 128]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=20, help='Length of sequence [default: 20]')
parser.add_argument('--num-digits', type=int, default=1, help='Number of moving digits [default: 1]')
parser.add_argument('--image-size', type=int, default=64, help='Image size [default: 64]')
#model parameters
parser.add_argument('--mode', type=str, default='basic', help='Basic model or advanced model [default: basic]')
parser.add_argument('--unit', type=str, default='graphrnn', help='Unit. graphrnn etc... [default: graphrnn]')
parser.add_argument('--step-length', type=float, default=0.1, help='Step length [default: 0.1]')
parser.add_argument('--alpha', type=float, default=1.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta', type=float, default= 1.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--alpha_color', type=float, default=0.0, help='Weigh on CD color loss [default: 1.0]')
parser.add_argument('--beta_color', type=float, default=0.0, help='Weigh on EMD color loss [default: 1.0]')
parser.add_argument('--log-dir', default='mmnist', help='Log dir [default: outputs/mminst]')
parser.add_argument('--version', default='v1', help='Model version')
parser.add_argument('--restore-training', type= int , default = 0 , help='restore-training [default: 0=False 1= True]')

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

# SET UP DIR
summary_dir = args.log_dir
summary_dir = 'summary_dir/Mnist/' + summary_dir 
summary_dir += '-%ddigit-%s-%s'%(args.num_digits, args.mode, args.unit)
summary_dir += '_'+ str(args.version)

args.log_dir = 'outputs/' +args.log_dir 
args.log_dir += '-%ddigit-%s-%s'%(args.num_digits, args.mode, args.unit)
args.log_dir += '_'+ str(args.version)

print("\n ==== GRAPH-RNN for MNIST DATASET  ====== \n")

#Define training dataset
train_dataset = MMNIST(root=args.data_dir,
                       seq_length=args.seq_length,
                       num_digits=args.num_digits,
                       image_size=args.image_size,
                       step_length=args.step_length,
                       num_points=args.num_points,
                       train=True)
print("[Dataset loaded]:", train_dataset)

# Function to get bacth
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
              alpha_color=args.alpha_color,
              beta_color=args.beta_color,
              learning_rate=args.learning_rate,
              max_gradient_norm=args.max_gradient_norm,
              activation = args.activation,
              is_training=True)

# Summary params    
tf.summary.scalar('cd', model.cd)
tf.summary.scalar('emd', model.emd)
tf.summary.scalar('loss', model.loss)


# Gradients summary if you wish to save them in the summary
for grad, var in model.gradients:
	tf.summary.histogram("%s_grads" % str(var.name), grad)
	grad_mean= tf.reduce_mean(tf.abs(grad))
	tf.summary.histogram("%s_weights" % str(var.name), var)
	tf.summary.scalar(var.name, grad_mean)
	print("var.name:", var.name)
	print("var:", var)

	
summary_op = tf.summary.merge_all()

# Checkpoint Directory
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')

# Restore Checkpoint
ckpt_number = 0
if (args.restore_training == True):
	checkpoint_path_automatic = tf.train.latest_checkpoint(checkpoint_dir)
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

# Allow dynamic allocation of computer resources
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(  config = config) as sess:
#with tf.Session( ) as sess:

    if args.restore_training == False:
    	sess.run(tf.global_variables_initializer())
    else:
    	print ("\n** RESTORE FROM CHECKPOINT ***\ncheckpoint_path_restore: ", checkpoint_path_automatic)
    	model.saver.restore(sess, checkpoint_path_automatic)    
   
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    
    for i in range(ckpt_number,args.num_iters):
        
        # LOAD DATA
        batch_data = get_batch(dataset=train_dataset, batch_size=args.batch_size)
        feed_dict = {model.inputs: batch_data}
        
        # RUN MODEL
        cd, emd, step, summary, predictions, _ = sess.run([model.cd, model.emd, model.global_step, summary_op, model.predicted_frames, model.train_op], feed_dict=feed_dict)
        print('[GEO LOSS] [%10d:]\t%.12f\t%.12f'%(i+1, cd, emd))
	
	# WRITE TO LOG
        log.write('[%s]\t[%10d:]\t%.12f\t%.12f\n'%(str(datetime.now()), i+1, cd, emd))
        log.flush()
        
        # WRITE SUMMARY
        if (i+1) % args.save_summary == 0:
        	summary_writer.add_summary(summary, step)
        
	# SAVE CHECKPOINT
        if (i+1) % args.save_cpk == 0:
            ckpt = os.path.join(checkpoint_path, )
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)        
        
        # SAVE EXAMPLES (with outlier cases)
        if  ( (i+1) % args.save_iters == 0 or  ( emd > 10.0 and i > 2000) or (i==0)):
            ckpt = os.path.join(checkpoint_path, )
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            pc_ground_truth = batch_data[0, int(args.seq_length/2):,:]
            pc_prediction = predictions[0]                         

            ground_truth = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))
            prediction = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))

            pc_ground_truth = np.ceil(pc_ground_truth).astype(np.uint8)
            pc_prediction = np.ceil(pc_prediction).astype(np.uint8)

            pc_prediction = np.clip(pc_prediction, a_min=0, a_max=args.image_size-1)

            for j in range(int(args.seq_length/2)):
                for k in range(args.num_points):
                    ground_truth[j, pc_ground_truth[j,k,0], pc_ground_truth[j,k,1]] = 255
                    prediction[j, pc_prediction[j,k,0], pc_prediction[j,k,1]] = 255
            ground_truth = np.swapaxes(ground_truth.astype(np.uint8), 0, 1)
            prediction = np.swapaxes(prediction.astype(np.uint8), 0, 1)
            ground_truth = np.reshape(ground_truth, (args.image_size, -1))
            prediction = np.reshape(prediction, (args.image_size, -1))
            image = np.concatenate((ground_truth, prediction), 0)              
            image[args.image_size, :] = 255
            for j in range(1, int(args.seq_length/2)):
                image[:, j*args.image_size] = 255
            image = Image.fromarray(image, 'L')
            image_path = os.path.join(example_dir, '%d.png'%step)
            image.save(image_path)

            buff = io.BytesIO()
            image.save(buff, format='PNG')
            image_string = buff.getvalue()
            buff.close()
            example = tf.Summary.Image(height=3*args.image_size, width=int(args.seq_length/2)*args.image_size, colorspace=1, encoded_image_string=image_string)
            summary = tf.Summary(value=[tf.Summary.Value(tag='image', image=example)])
            summary_writer.add_summary(summary, step)
log.close()


