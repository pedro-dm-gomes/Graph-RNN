"""
Script to Evaluate a Graph-RNN model 
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
from sklearn.decomposition import PCA
# Load Long-Term Model
import  models.GraphRNN_LongTerm_models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

parser = argparse.ArgumentParser()

# 1 digit 
parser.add_argument('--data-path', default='/home/uceepdg/profile.V6/Desktop/MNIST/test-1mnist-64-128point-20step.npy', help='Data path [default: data/test-1mnist-64-128point-20step.npy]')

parser.add_argument('--ckpt-step', type=int, default=200000, help='Manual Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=128, help='Number of points [default: 128]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=20, help='Length of sequence [default: 20]')
parser.add_argument('--num-digits', type=int, default=1, help='Number of moving digits [default: 1]')

# 2 digits 
"""
parser.add_argument('--data-path', default='/home/uceepdg/profile.V6/Desktop/MNIST/test-2mnist-64-256point-20step.npy', help='Data path [default: data/test-2mnist-64-256point-20step.npy]')
parser.add_argument('--ckpt-step', type=int, default=800000, help='Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=128, help='Number of points [default: 256]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples[default: 4]')
parser.add_argument('--seq-length', type=int, default=20, help='Length of sequence [default: 20]')
parser.add_argument('--num-digits', type=int, default= 1, help='Number of moving digits [default: 1]')
"""

parser.add_argument('--mode', type=str, default='basic', help='Basic model or advanced model [default: advanced]')
parser.add_argument('--unit', type=str, default='graphrnn', help='Unit. graphrnn [default: graphrnn]')
parser.add_argument('--activation', type= int, default=0, help=' Activation function [default: 0=None or 1=tf.nn.relu]')

parser.add_argument('--image-size', type=int, default=64, help='Image size [default: 64]')
parser.add_argument('--log-dir', default='mmnist', help='Log dir [default: outputs/mminst]')
parser.add_argument('--version', default='v1', help='Model version')

print("\n EVALUATION SCRIPT \n")

args = parser.parse_args()

data = np.load(args.data_path)
data = np.concatenate((data, np.zeros((data.shape[0], args.seq_length, args.num_points, 1), dtype=data.dtype)),3)
gt_data = np.load(args.data_path)
n_pcs, seq_len, n_pts, dim = gt_data.shape

# SET UP DIR
summary_dir = args.log_dir
summary_dir = 'summary_dir/Mnist/' + summary_dir 
summary_dir += '-%ddigit-%s-%s'%(args.num_digits, args.mode, args.unit)
summary_dir += '_'+ str(args.version)
print("summary dir: ",summary_dir)

args.log_dir = 'outputs/' +args.log_dir 
args.log_dir += '-%ddigit-%s-%s'%(args.num_digits, args.mode, args.unit)
args.log_dir += '_'+ str(args.version)
print("ouput dir: ",args.log_dir)

# Set activation function (work-around parser)
if(args.activation == 0):
	args.activation = None
if(args.activation == 1):
	args.activation = tf.nn.relu

	
model_name = args.mode.capitalize() + 'Graph' + args.unit[5:].upper() 
print("Call model: ", model_name)
Model = getattr(models, model_name)
model = Model(1,
              num_points=args.num_points,
              num_samples=args.num_samples,
              seq_length=args.seq_length,
              knn=True,
              activation= args.activation,
              is_training=False)
              

# Checkpoint Directory
checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')

# Restore Checkpoint
ckpt_number = 0
checkpoint_path_automatic = tf.train.latest_checkpoint(checkpoint_dir)
ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
ckpt_number=ckpt_number[5:]
ckpt_number=int(ckpt_number)

# Test Example Folder
example_dir = os.path.join(args.log_dir, 'test-examples')
example_dir = os.path.join(example_dir, '%04d'%(ckpt_number))
if not os.path.exists(example_dir):
    os.makedirs(example_dir)
    
test_dir = os.path.join(example_dir, 'PCA')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

#Evaluation  Log    
log = open(os.path.join(args.log_dir, 'eval_ckp_' + str(ckpt_number) +'.log'), 'w')
#write input arguments
log.write("\n ========  Evaluation Log ========  \n")
log.write(":Input Arguments\n")
for var in args.__dict__:
	log.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )
log.flush()


with tf.Session() as sess:
    
    # Restore Model
    model.saver.restore(sess, checkpoint_path_automatic)

    flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    log.write ('\n flops: [%s]'% (flops))
    log.write ('\n parameters: [%s]'% (parameters))
    log.write ('\ntotal flops: {}'.format(flops.total_float_ops))
    log.write ('\ntotal parameters: {}'.format(parameters.total_parameters))
    log.flush()
    
    outputs = []
    t_cd = t_emd =0

    for i in range(data.shape[0]):


        if(i %(100) ==0):
        	print("[",i," /",data.shape[0],"]")
        	print("[%d] Total [CD]: %f\t [EMD]: %f\t \n"%(i, t_cd/n_pcs, t_emd/n_pcs ) ) 
        	
        sequence_nr =i
        curr_dir = os.path.join(example_dir, '%04d'%(i+1))
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        batch_data = np.expand_dims(data[i], axis=0)

        # feed only one example
        feed_dict = {model.inputs: batch_data}
        
        # Define my inputs
        out = []
        inputs = [
        	model.predicted_frames,
        	model.downsample_frames,
        	model.predicted_motions,
        	model.loss,
        	model.emd,
        	model.cd,
        	model.out_s_xyz1,
        	#model.out_s_color1,
        	model.out_s_feat1,
        	model.out_s_states1,
        	model.out_s_xyz2,
        	#model.out_s_color2,
        	model.out_s_feat2,
        	model.out_s_states2,
        	model.out_s_xyz3,
        	#model.out_s_color3,
        	model.out_s_feat3,
        	model.out_s_states3
        	]
             		        		
        # Run Session 
        out = sess.run( inputs, feed_dict=feed_dict)
        
        [predicted_frames,
        downsample_frames,
        predicted_motions,
        loss,
        emd,
        cd,
        out_s_xyz1,
        #out_s_color1,
        out_s_feat1,
        out_s_states1,
        out_s_xyz2,
        #out_s_color2,        
        out_s_feat2,
        out_s_states2,
        out_s_xyz3,
        #out_s_color3,
        out_s_feat3,
        out_s_states3 ] = out
        
        outputs.append(predicted_frames)
        
        """ Save PNG RESULTS """
        pc_context = batch_data[0, :int(args.seq_length/2),:]
        pc_ground_truth = batch_data[0, int(args.seq_length/2):,:]
        pc_prediction = predicted_frames[0]                                                  # [int(args.seq_length/2), num_digits, 3]
        
        context = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))
        ground_truth = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))
        prediction = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))

        pc_context = np.ceil(pc_context).astype(np.uint8)
        pc_ground_truth = np.ceil(pc_ground_truth).astype(np.uint8)
        pc_prediction = np.ceil(pc_prediction).astype(np.uint8)

        pc_prediction = np.clip(pc_prediction, a_min=0, a_max=args.image_size-1)

        for j in range(int(args.seq_length/2)):
            for k in range(args.num_points):
                context[j, pc_context[j,k,0], pc_context[j,k,1]] = 255
                ground_truth[j, pc_ground_truth[j,k,0], pc_ground_truth[j,k,1]] = 255
                prediction[j, pc_prediction[j,k,0], pc_prediction[j,k,1]] = 255
        context = np.swapaxes(context.astype(np.uint8), 0, 1)
        ground_truth = np.swapaxes(ground_truth.astype(np.uint8), 0, 1)
        prediction = np.swapaxes(prediction.astype(np.uint8), 0, 1)

        context = np.reshape(context, (args.image_size, -1))
        ground_truth = np.reshape(ground_truth, (args.image_size, -1))
        prediction = np.reshape(prediction, (args.image_size, -1))

        for j in range(1, int(args.seq_length/2)):
            context[:, j*args.image_size] = 255
            ground_truth[:, j*args.image_size] = 255
            prediction[:, j*args.image_size] = 255

        context = Image.fromarray(context, 'L')
        ground_truth = Image.fromarray(ground_truth, 'L')
        prediction = Image.fromarray(prediction, 'L')

        context.save(os.path.join(curr_dir, 'ctx.png'))
        ground_truth.save(os.path.join(curr_dir, 'gth.png'))
        prediction.save(os.path.join(curr_dir, 'pdt.png'))
        
        
        # Print ERROR
        t_cd = t_cd + cd
        t_emd = t_emd + emd
        t_pcs = int(i)
        
    
    outputs = np.concatenate(outputs, 0)
    np.save(os.path.join(args.log_dir, 'test-predictions'), outputs)        
            
    # Save to File
    print(" Final Values ")
    print("Total emd", t_emd)
    print("Total cd", t_cd)
    print("AVG CD/n_pcs ",(t_cd/t_pcs) )
    print("AVG EMD/n_pcs ",(t_emd/t_pcs) )
    print("AVG EMP /(n_pcs * n_pts) )",(t_emd/(t_pcs *n_pts) ) )  
    

    #log = open(os.path.join(args.log_dir, 'cd_emd_ckp_' + str(ckpt_number) +'.log'), 'w')
    log.write('Total CD [%f]\t Total EMD[%f]\n'%( t_cd, t_emd)  )
    log.write('AVG CD [%f]\t AVG EMD[%f]\n'%( t_cd/n_pcs, t_emd/n_pcs )  )
    log.write('AVG CD/n_pts [%f]\t AVG EMD/n_pts[%f]\n'%( t_cd/(n_pcs*n_pts), t_emd/(n_pcs*n_pts))  )
    log.flush()    

