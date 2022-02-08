import os
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
import models.GraphRNN_LongTerm_models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))

from pointnet2_color_feat_states import *
from graphrnn_cell import *
import tf_nndistance
import tf_approxmatch

import tf_nndistance
import tf_approxmatch
import tf_util

#Convert to 
nr_points= 1000

""" YOU MUST SELECT THE FOLDERS """
input_points_dir = '/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys/test/4000'
input_color_dir = '/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys_Color/test/4000'

output_points_dir =os.path.join('/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys/test/', str(nr_points))
output_color_dir =os.path.join('/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys_Color/test/', str(nr_points))

if not os.path.exists(output_points_dir):
	os.makedirs(output_points_dir)
if not os.path.exists(output_color_dir):
	os.makedirs(output_color_dir)    
    
print("input_points_dir",input_points_dir)
print("input_color_dir",input_color_dir)
print("output_points_dir",output_points_dir)
print("output_color_dir",output_color_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(  config = config) as sess:


	print("\n SESSISON")
	#Go To directory
	for character in sorted (os.listdir(input_points_dir)):
		
		print("\n\nCharacter: ", character)
		character_path = os.path.join(input_points_dir, character)
		character_path_color = os.path.join(input_color_dir, character)
		
		if(character !='Louised'):
			for sequence in sorted(os.listdir(character_path), reverse =False):
				print("\n[",character,"] - ", sequence)
				sequence_path = os.path.join(character_path, sequence)
				sequence_path_color = os.path.join(character_path_color, sequence)
				
				seq_points =[]
				for npy in sorted(os.listdir(sequence_path)):
					npy_file = os.path.join(sequence_path, npy)
					print("npy_file:",npy_file)
					npy_data = np.load(npy_file)
					#npy_data=np.ones( (10,3) )
					print("npy_data",npy_data.shape)
					seq_points.append(npy_data)
				
				seq_color =[]
				for npy in sorted(os.listdir(sequence_path_color)):
					npy_file = os.path.join(sequence_path_color, npy)
					print("npy_file:",npy_file)
					npy_data = np.load(npy_file)
					#npy_data=np.ones( (10,3) )
					print("npy_data",npy_data.shape)
					seq_color.append(npy_data)			
		

				seq_points = np.array(seq_points)
				seq_color = np.array(seq_color)
				print("seq_points.shape",seq_points.shape)
				print("seq_color.shape",seq_color.shape)
				
				print("  [Downsample op] ")
				
				length = seq_points.shape[0]
				
				for frame in range(0,length):
					print("frame[",frame,"]")

					xyz = seq_points[frame]
					color =seq_color[frame] # ERROR GRAVE
					xyz =np.expand_dims(xyz, 0)
					color =np.expand_dims(color, 0)
					print("xyz.shape",xyz.shape)
					print("color.shape",color.shape)					
					
					xyz,color, feat, _, _,_ = sample_and_group(int(nr_points), radius=0.1+1e-20, nsample=1, xyz=xyz,color=color, features=None,  states =None,  knn=False, use_xyz=False)
							
					xyz = np.array( sess.run(xyz))
					color = np.array( sess.run(color))
					
					xyz= xyz[0]
					color =color[0]

					print("xyz.shape",xyz.shape)
					print("color.shape",color.shape)
					if (color.shape != xyz.shape):
						color = np.reshape(color, xyz.shape)
					print("color.shape",color.shape)
					print("color[2] ",color[2])
					# Save Frame
					if( frame <9) :
						npy_frame = 'frame_00000' + str(frame+1) + '.npy'
					else:
						npy_frame = 'frame_0000' + str(frame+1) + '.npy'
					save_points_dir =os.path.join(output_points_dir,character)
					save_points_dir =os.path.join(save_points_dir,sequence)

					save_color_dir =os.path.join(output_color_dir,character)
					save_color_dir =os.path.join(save_color_dir,sequence)
					print("save_points_dir",save_points_dir)
					print("save_color_dir",save_color_dir)
					if not os.path.exists(save_points_dir):
						os.makedirs(save_points_dir)
					if not os.path.exists(save_color_dir):				
						os.makedirs(save_color_dir)								
					save_points_dir=os.path.join(save_points_dir,npy_frame)
					save_color_dir =os.path.join(save_color_dir,npy_frame)						
						
					np.save(save_points_dir, xyz)   
					np.save(save_color_dir, color)  				
	        
					
			
			
			
   
