import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))


from pointnet2_color_feat_states import *
from graphrnn_cell_without_color import *

import tf_nndistance
import tf_approxmatch
import tf_util


class AdvancedGraphRNN(object):
    def __init__(self, batch_size, seq_length, num_points=256, num_samples=8, knn=False, alpha=1.0, beta=1.0,alpha_color=0.0, beta_color=0.0, learning_rate=0.001, max_gradient_norm=5.0, is_training=False):

        self.global_step = tf.Variable(0, trainable=False)

	# No color
        self.inputs = tf.placeholder(tf.float32, [batch_size, seq_length, num_points, 3])
        frames = tf.split(value=self.inputs, num_or_size_splits=seq_length, axis=1)
        frames = [tf.squeeze(input=frame, axis=[1]) for frame in frames]

        sampled_points = num_points
        sampled_points_down1 = int(sampled_points /2)
        sampled_points_down2 = int(sampled_points /2/2)
        sampled_points_down3=   int(sampled_points /2/2/2)
        
        context_frames = int(seq_length/2) # 10 frames
        
        cell_feat_1 = PointFeatureCell(radius=1.0+1e-6, nsample=2*num_samples, out_channels=64, knn=knn, pooling='max')
        cell_feat_2 = PointFeatureCell(radius=1.0+1e-8, nsample=2*num_samples, out_channels=128, knn=knn, pooling='max')
        cell_feat_3 = PointFeatureCell(radius=1.0+1e-12, nsample=1*num_samples, out_channels=128, knn=knn, pooling='max')
        
        graph_cell1 = GraphRNNCell_2N_1MAX(radius= 0.1, nsample=num_samples, out_channels=256, knn= True, pooling='max')
        graph_cell2 = GraphRNNCell_2N_1MAX(radius= 0.1, nsample=num_samples, out_channels=256, knn= True, pooling='max')
        graph_cell3 = GraphRNNCell_2N_1MAX(radius= 0.1, nsample=num_samples, out_channels=256, knn= True, pooling='max')
        
        
        print("batch_size:",batch_size)
        print("seq_length:",seq_length)
        print("context_frames:",context_frames)
        print("num_points:",num_points)
        print("inputs:",self.inputs)
        print("sampled_points:",sampled_points)
        print("alpha_color:",alpha_color)
        print("beta_color:",beta_color)
        print("inputs:",self.inputs)
        
        # STATES
        global_state1 = None
        global_state2 = None
        global_state3 = None
        
        
        # prediction
        predicted_motions = []
        predicted_motions_colors = []
        predicted_frames = []
        downsample_frames = []
        
        #output_States
        self.out_s_xyz1 =[]
        self.out_s_color1 =[]       
        self.out_s_feat1=[]
        self.out_s_states1 =[]
        self.out_s_xyz2 =[]
        self.out_s_color2 =[]       
        self.out_s_feat2=[]
        self.out_s_states2 =[]
        self.out_s_xyz3 =[]
        self.out_s_color3 =[]       
        self.out_s_feat3=[]
        self.out_s_states3 =[]
        
        #save neighbor of central point 100
        self.out_nbrs1 =[]
        self.out_nbrs2 =[]
        self.out_nbrs3 =[]
        self.out_cp1 =[]
        self.out_cp2 =[]
        self.out_cp3 =[]
        
        #output_state_propagation
        self.out_l2_feat =[]
        self.out_l1_feat =[]
        self.out_l0_feat =[]
        
        self.extra =[]

        input_frame = frames[0]
        

        
        print(" ========= CONTEXT  ============")
        for i in range(int(context_frames) ):

            print("contex frames down[",i, "]")
            input_frame = frames[i]
            input_frame_points = input_frame
            xyz0 = input_frame_points
                
            
            print("\n === Downsample Module 1  ====") 
            xyz1, color1, feat1, states1, _, _ = sample_and_group(int(sampled_points_down1), radius=1.0+1e-8, nsample= 4, xyz=xyz0,  color=input_frame_points, features=None, states = None, knn=True, use_xyz=False)            

                        
            print("\n === CELL 1  Point-Features ====")
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE) as scope:
            	out_1 = cell_feat_1((xyz1, None, None, None))
            	f_xyz1, f_color1, f_feat1, f_states1 = out_1
            	print("f_xyz1",f_xyz1)
            	print("f_feat1",f_feat1)
            	print("f_color1",f_color1)
            	print("f_states1",f_states1)
            	print("\n")
            
            print("\n === CELL 2  Point-Features ====")
            with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE) as scope:
            	out_2 = cell_feat_2((f_xyz1, None, f_feat1, None))
            	f_xyz2, f_color2, f_feat2, f_states2 = out_2
            	print("f_xyz2",f_xyz2)
            	print("f_feat2",f_feat2)
            	print("f_color2",f_color2)
            	print("f_states2",f_states2)
            	print("\n")
            	
            print("\n === CELL 3  Point-Features ====")
            with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE) as scope:
            	out_3 = cell_feat_3((f_xyz2, None, f_feat2, None))
            	f_xyz3, f_color3, f_feat3, f_states3 = out_3
            	print("f_xyz3",f_xyz3)
            	print("f_feat3",f_feat3)
            	print("f_color3",f_color3)
            	print("f_states3",f_states3)
            	print("\n")   
            	            
         
            #create point time
            time = tf.fill( (f_xyz3.shape[0],f_xyz3.shape[1],1), (i/1.0))
            print("time.shape", time)

            print("\n === CELL 1  GraphRNN group by 2 Neighborhoods by features  ====") 
            with tf.variable_scope('decoder_1', reuse=tf.AUTO_REUSE) as scope:
            	global_state1 = graph_cell1( (f_xyz3, None, f_feat3, None, time), global_state1)
            	s_xyz1, s_color1, s_feat1, s_states1, time,nbrs1, cp1, extra  = global_state1
            	print("s_xyz1",s_xyz1)
            	print("s_feat1",s_feat1)
            	print("s_color1",s_color1)
            	print("s_states1",s_states1)
            	print("\n")
            
            print("\n === CELL 2 GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz2, color2, feat2, states2, _, _ = sample_and_group(int(sampled_points_down2), radius=1.0+1e-20, nsample= 4 , xyz=s_xyz1,  color=s_xyz1, features=s_feat1, states = s_states1, knn=True, use_xyz=False)                
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            states2 = tf.reduce_max(states2, axis=[2], keepdims=False, name='maxpool')
            time = tf.fill( (xyz2.shape[0],xyz2.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_2', reuse=tf.AUTO_REUSE) as scope:
                global_state2 = graph_cell2( (xyz2, None, feat2, states2, time), global_state2)
                s_xyz2, s_color2, s_feat2, s_states2, time,nbrs2, cp2,_ = global_state2
                print("s_xyz2",s_xyz2)
                print("s_feat2",s_feat2)
                print("s_color2",s_color2)
                print("s_states2",s_states2)
                print("\n")                

            print("\n === CELL 3  GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz3, color3, feat3, states3, _, _ = sample_and_group(int(sampled_points_down3), radius=4.0+1e-20, nsample= 4, xyz=s_xyz2,  color=s_xyz2, features=s_feat2, states = s_states2, knn=True, use_xyz=False)                
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            states3 = tf.reduce_max(states3, axis=[2], keepdims=False, name='maxpool')
            time = tf.fill( (xyz3.shape[0],xyz3.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_3', reuse=tf.AUTO_REUSE) as scope:
                global_state3 = graph_cell3( (xyz3, None, feat3, states3, time), global_state3)
                s_xyz3, s_color3,s_feat3, s_states3, time,nbrs3, cp3,_  = global_state3
                print("s_xyz3",s_xyz3)
                print("s_feat3",s_feat3)
                print("s_color3",s_color3)
                print("s_states3",s_states3)                
                print("\n")

        
        
        print(" ========= Prediction   ============")
        
        input_frame = frames[int(seq_length/2)-1]
        # input frame = frame [9]
        
        for i in range(int(context_frames), seq_length  ):

            print("frame [",i-1,"]  predicts -> [",i -(context_frames),"]")
            
            
            input_frame_points = input_frame
            input_frame_color = input_frame
            xyz0 = input_frame_points
            
            
            print("\n === Downsample Module 1  ====") 
            xyz1, color1, feat1, states1, _, _ = sample_and_group(int(sampled_points_down1), radius=1.0+1e-8, nsample= 4, xyz=input_frame_points,  color=input_frame_color, features=None, states = None, knn=True, use_xyz=False)            
            feat1 = tf.reduce_max(feat1, axis=[2], keepdims=False, name='maxpool')
            states1 = tf.reduce_max(states1, axis=[2], keepdims=False, name='maxpool')
                        
            print("\n === CELL 1  Point-Features ====")
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE) as scope:
            	out_1 = cell_feat_1((xyz1, None, None, None))
            	f_xyz1, f_color1, f_feat1, f_states1 = out_1
            	print("f_xyz1",f_xyz1)
            	print("f_feat1",f_feat1)
            	print("f_color1",f_color1)
            	print("f_states1",f_states1)
            	print("\n")
            
            print("\n === CELL 2  Point-Features ====")
            with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE) as scope:
            	out_2 = cell_feat_2((f_xyz1, None, f_feat1, None))
            	f_xyz2, f_color2, f_feat2, f_states2 = out_2
            	print("f_xyz2",f_xyz2)
            	print("f_feat2",f_feat2)
            	print("f_color2",f_color2)
            	print("f_states2",f_states2)
            	print("\n")
            	
            print("\n === CELL 3  Point-Features ====")
            with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE) as scope:
            	out_3 = cell_feat_3((f_xyz2, None, f_feat2, None))
            	f_xyz3, f_color3, f_feat3, f_states3 = out_3
            	print("f_xyz3",f_xyz3)
            	print("f_feat3",f_feat3)
            	print("f_color3",f_color3)
            	print("f_states3",f_states3)
            	print("\n")   
            	            
         
            #create point time
            time = tf.fill( (f_xyz3.shape[0],f_xyz3.shape[1],1), (i/1.0))
            print("time.shape", time)

            print("\n === CELL 1  GraphRNN group by 2 Neighborhoods by features  ====") 
            with tf.variable_scope('decoder_1', reuse=tf.AUTO_REUSE) as scope:
            	global_state1 = graph_cell1( (f_xyz3, None, f_feat3, None, time), global_state1)
            	s_xyz1, s_color1, s_feat1, s_states1, time,nbrs1, cp1, extra  = global_state1
            	print("s_xyz1",s_xyz1)
            	print("s_feat1",s_feat1)
            	print("s_color1",s_color1)
            	print("s_states1",s_states1)
            	print("\n")
            
            print("\n === CELL 2 GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz2, color2, feat2, states2, _, _ = sample_and_group(int(sampled_points_down2), radius=1.0+1e-20, nsample= 4 , xyz=s_xyz1,  color=s_xyz1, features=s_feat1, states = s_states1, knn=True, use_xyz=False)                
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            states2 = tf.reduce_max(states2, axis=[2], keepdims=False, name='maxpool')
            time = tf.fill( (xyz2.shape[0],xyz2.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_2', reuse=tf.AUTO_REUSE) as scope:
                global_state2 = graph_cell2( (xyz2, None, feat2, states2, time), global_state2)
                s_xyz2, s_color2, s_feat2, s_states2, time,nbrs2, cp2,_ = global_state2
                print("s_xyz2",s_xyz2)
                print("s_feat2",s_feat2)
                print("s_color2",s_color2)
                print("s_states2",s_states2)
                print("\n")                

            print("\n === CELL 3  GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz3, color3, feat3, states3, _, _ = sample_and_group(int(sampled_points_down3), radius=4.0+1e-20, nsample= 4, xyz=s_xyz2,  color=s_xyz2, features=s_feat2, states = s_states2, knn=True, use_xyz=False)                
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            states3 = tf.reduce_max(states3, axis=[2], keepdims=False, name='maxpool')
            time = tf.fill( (xyz3.shape[0],xyz3.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_3', reuse=tf.AUTO_REUSE) as scope:
                global_state3 = graph_cell3( (xyz3, None, feat3, states3, time), global_state3)
                s_xyz3, s_color3,s_feat3, s_states3, time,nbrs3, cp3,_  = global_state3
                print("s_xyz3",s_xyz3)
                print("s_feat3",s_feat3)
                print("s_color3",s_color3)
                print("s_states3",s_states3)                
                print("\n")

            with tf.variable_scope('fp', reuse=tf.AUTO_REUSE) as scope:
                l2_feat = pointnet_fp_module_original(xyz2,
                                             xyz3,
                                             s_states2,
                                             s_states3,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp2')
                l1_feat = pointnet_fp_module_original(xyz1,
                                             xyz2,
                                             s_states1,
                                             l2_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp1')
                l0_feat = pointnet_fp_module_original(xyz0,
                                             xyz1,
                                             None,
                                             l1_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp0')

                        
            with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
                predicted_motion = tf.layers.conv1d(inputs=l0_feat, filters=128, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc1')
                predicted_motion = tf.layers.conv1d(inputs=predicted_motion, filters=3, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc2')
                
         
            predicted_motions.append(predicted_motion)
            predicted_frame = xyz0 + predicted_motion      
            predicted_frames.append(predicted_frame)
                 
            #input is prediction
            input_frame = input_frame + predicted_motion
            
            self.extra.append(extra)
            self.out_s_xyz1.append(s_xyz1)
            self.out_s_color1.append(s_color1)
            self.out_s_feat1.append(s_feat1)
            self.out_s_states1.append(s_states1)
            self.out_s_xyz2.append(s_xyz2)
            self.out_s_color2.append(s_color2)
            self.out_s_feat2.append(s_feat2)
            self.out_s_states2.append(s_states2)
            self.out_s_xyz3.append(s_xyz3)
            self.out_s_color3.append(s_color3)
            self.out_s_feat3.append(s_feat3)
            self.out_s_states3.append(s_states3)
            self.out_l2_feat.append(l2_feat)
            self.out_l1_feat.append(l1_feat)
            self.out_l0_feat.append(l0_feat)
            self.out_nbrs1.append(nbrs1)
            self.out_nbrs2.append(nbrs2)
            self.out_nbrs3.append(nbrs3)
            self.out_cp1.append(cp1)
            self.out_cp2.append(cp2)
            self.out_cp3.append(cp3)            
            
        
        downsample_frames = frames                
        self.downsample_frames = downsample_frames
        self.predicted_motions = predicted_motions
        
        print("frames.shape", np.shape(frames))
        print("downsample_frames.shape", np.shape(downsample_frames) )
        print("predicted_frames.shape", np.shape(predicted_frames) )

        self.loss = self.emd = self.cd = 0
        self.emd_color = self.cd_color = 0
        self.diff =0
        self.frame_diff=[]
        self.frame_loss_cd=[]
        self.frame_loss_emd=[]

        for i in range(int(seq_length/2)):

            	# Select and split frames
            	print("downsample_frames[",i+int(seq_length/2),"] compare with -> predicted_frames[",i,"]")
            	frame = downsample_frames[i+int(seq_length/2)]
            	predicted_frame =predicted_frames[i]
            	frame_points = frame
            	predicted_frame_points = predicted_frame

            	# Geometry loss
            	match = tf_approxmatch.approx_match(frame_points,predicted_frame_points)
            	match_cost = tf_approxmatch.match_cost(frame_points, predicted_frame_points, match)
            	emd_distance = tf.reduce_mean(match_cost)
            	loss_emd = emd_distance
            	self.emd += loss_emd
            	self.frame_loss_emd.append(loss_emd)            	
            	
            	dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(predicted_frame_points, frame_points)
            	loss_cd = tf.reduce_mean(dists_forward+dists_backward)
            	self.cd += loss_cd
            	self.frame_loss_cd.append(loss_cd)    
            	
            	diff = abs(dists_forward) + abs(dists_backward)
            	self.frame_diff.append(diff)

            	
            	#Global Loss
            	self.loss += ( alpha*(loss_cd)  + (beta*loss_emd) )

        self.cd /= int(seq_length/2)  
        self.emd /= (int(seq_length/2)*num_points)

        
        self.loss /= int(seq_length/2)

        if is_training == True :
        
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.predicted_motions = tf.stack(values=predicted_motions, axis=1)
        self.predicted_frames = tf.stack(values=predicted_frames, axis=1)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        if is_training == False:
        
        	print("Dont update paramenters")
        	params = tf.trainable_variables()
        	#print("params",params)

class BasicGraphRNN(object):
    def __init__(self, batch_size, seq_length, num_points=256, num_samples=8, knn=False, alpha=1.0, beta=1.0,alpha_color=0.0, beta_color=0.0, learning_rate=0.001, max_gradient_norm=5.0, is_training=False):

        self.global_step = tf.Variable(0, trainable=False)

	# No color
        self.inputs = tf.placeholder(tf.float32, [batch_size, seq_length, num_points, 3])
        frames = tf.split(value=self.inputs, num_or_size_splits=seq_length, axis=1)
        frames = [tf.squeeze(input=frame, axis=[1]) for frame in frames]


        context_frames = int(seq_length/2) # 10 frames
        
        cell_feat_1 = PointFeatureCell(radius=1.0+1e-6, nsample=2*num_samples, out_channels=64, knn=knn, pooling='max')
        cell_feat_2 = PointFeatureCell(radius=1.0+1e-8, nsample=2*num_samples, out_channels=128, knn=knn, pooling='max')
        cell_feat_3 = PointFeatureCell(radius=1.0+1e-12, nsample=1*num_samples, out_channels=128, knn=knn, pooling='max')
        
        graph_cell1 = GraphRNNCell_2N_1MAX(radius= 0.1, nsample=num_samples, out_channels=256, knn= True, pooling='max')
        graph_cell2 = GraphRNNCell_2N_1MAX(radius= 0.1, nsample=num_samples, out_channels=256, knn= True, pooling='max')
        graph_cell3 = GraphRNNCell_2N_1MAX(radius= 0.1, nsample=num_samples, out_channels=256, knn= True, pooling='max')
        
        
        print("batch_size:",batch_size)
        print("seq_length:",seq_length)
        print("context_frames:",context_frames)
        print("num_points:",num_points)
        print("inputs:",self.inputs)
        #print("sampled_points:",sampled_points)
        print("alpha_color:",alpha_color)
        print("beta_color:",beta_color)
        print("inputs:",self.inputs)
        
        # STATES
        global_state1 = None
        global_state2 = None
        global_state3 = None
        
        
        # prediction
        predicted_motions = []
        predicted_motions_colors = []
        predicted_frames = []
        downsample_frames = []
        
        #output_States
        self.out_s_xyz1 =[]
        self.out_s_color1 =[]       
        self.out_s_feat1=[]
        self.out_s_states1 =[]
        self.out_s_xyz2 =[]
        self.out_s_color2 =[]       
        self.out_s_feat2=[]
        self.out_s_states2 =[]
        self.out_s_xyz3 =[]
        self.out_s_color3 =[]       
        self.out_s_feat3=[]
        self.out_s_states3 =[]
        
        #save neighbor of central point 100
        self.out_nbrs1 =[]
        self.out_nbrs2 =[]
        self.out_nbrs3 =[]
        self.out_cp1 =[]
        self.out_cp2 =[]
        self.out_cp3 =[]
        
        #output_state_propagation
        self.out_l2_feat =[]
        self.out_l1_feat =[]
        self.out_l0_feat =[]
        
        self.extra =[]

        input_frame = frames[0]
        
        print(" ========= CONTEXT  ============")
        for i in range(int(context_frames) ):

            print("contex frames down[",i, "]")
            input_frame = frames[i]
            input_frame_points = input_frame
            xyz0 = input_frame_points
      

                        
            print("\n === CELL 1  Point-Features ====")
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE) as scope:
            	out_1 = cell_feat_1((input_frame, None, None, None))
            	f_xyz1, f_color1, f_feat1, f_states1 = out_1
            	print("f_xyz1",f_xyz1)
            	print("f_feat1",f_feat1)
            	print("f_color1",f_color1)
            	print("f_states1",f_states1)
            	print("\n")
            
            print("\n === CELL 2  Point-Features ====")
            with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE) as scope:
            	out_2 = cell_feat_2((f_xyz1, None, f_feat1, None))
            	f_xyz2, f_color2, f_feat2, f_states2 = out_2
            	print("f_xyz2",f_xyz2)
            	print("f_feat2",f_feat2)
            	print("f_color2",f_color2)
            	print("f_states2",f_states2)
            	print("\n")
            	
            print("\n === CELL 3  Point-Features ====")
            with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE) as scope:
            	out_3 = cell_feat_3((f_xyz2, None, f_feat2, None))
            	f_xyz3, f_color3, f_feat3, f_states3 = out_3
            	print("f_xyz3",f_xyz3)
            	print("f_feat3",f_feat3)
            	print("f_color3",f_color3)
            	print("f_states3",f_states3)
            	print("\n")   
            	            
         
            #create point time
            time = tf.fill( (f_xyz3.shape[0],f_xyz3.shape[1],1), (i/1.0))
            print("time.shape", time)

            print("\n === CELL 1  GraphRNN group by 2 Neighborhoods by features  ====") 
            with tf.variable_scope('decoder_1', reuse=tf.AUTO_REUSE) as scope:
            	global_state1 = graph_cell1( (f_xyz3, None, f_feat3, None, time), global_state1)
            	s_xyz1, s_color1, s_feat1, s_states1, time,nbrs1, cp1, extra  = global_state1
            	print("s_xyz1",s_xyz1)
            	print("s_feat1",s_feat1)
            	print("s_color1",s_color1)
            	print("s_states1",s_states1)
            	print("\n")
            
            print("\n === CELL 2 GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz2 =   s_xyz1        
            feat2 =  s_feat1
            states2 = s_states1
            time = tf.fill( (xyz2.shape[0],xyz2.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_2', reuse=tf.AUTO_REUSE) as scope:
                global_state2 = graph_cell2( (xyz2, None, feat2, states2, time), global_state2)
                s_xyz2, s_color2, s_feat2, s_states2, time,nbrs2, cp2,_ = global_state2
                print("s_xyz2",s_xyz2)
                print("s_feat2",s_feat2)
                print("s_color2",s_color2)
                print("s_states2",s_states2)
                print("\n")                

            print("\n === CELL 3  GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz3 =   s_xyz2        
            feat3 =  s_feat2
            states3 = s_states2
            time = tf.fill( (xyz3.shape[0],xyz3.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_3', reuse=tf.AUTO_REUSE) as scope:
                global_state3 = graph_cell3( (xyz3, None, feat3, states3, time), global_state3)
                s_xyz3, s_color3,s_feat3, s_states3, time,nbrs3, cp3,_  = global_state3
                print("s_xyz3",s_xyz3)
                print("s_feat3",s_feat3)
                print("s_color3",s_color3)
                print("s_states3",s_states3)                
                print("\n")

        
        
        print(" ========= Prediction   ============")
        
        input_frame = frames[int(seq_length/2)-1]
        # input frame = frame [9]
        
        for i in range(int(context_frames), seq_length  ):

            print("frame [",i-1,"]  predicts -> [",i -(context_frames),"]")
            
            input_frame_points = input_frame
            input_frame_color = input_frame
            xyz0 = input_frame_points
            
            print("\n === CELL 1  Point-Features ====")
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE) as scope:
            	out_1 = cell_feat_1((input_frame, None, None, None))
            	f_xyz1, f_color1, f_feat1, f_states1 = out_1
            	print("f_xyz1",f_xyz1)
            	print("f_feat1",f_feat1)
            	print("f_color1",f_color1)
            	print("f_states1",f_states1)
            	print("\n")
            
            print("\n === CELL 2  Point-Features ====")
            with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE) as scope:
            	out_2 = cell_feat_2((f_xyz1, None, f_feat1, None))
            	f_xyz2, f_color2, f_feat2, f_states2 = out_2
            	print("f_xyz2",f_xyz2)
            	print("f_feat2",f_feat2)
            	print("f_color2",f_color2)
            	print("f_states2",f_states2)
            	print("\n")
            	
            print("\n === CELL 3  Point-Features ====")
            with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE) as scope:
            	out_3 = cell_feat_3((f_xyz2, None, f_feat2, None))
            	f_xyz3, f_color3, f_feat3, f_states3 = out_3
            	print("f_xyz3",f_xyz3)
            	print("f_feat3",f_feat3)
            	print("f_color3",f_color3)
            	print("f_states3",f_states3)
            	print("\n")   
            	            
         
            #create point time
            time = tf.fill( (f_xyz3.shape[0],f_xyz3.shape[1],1), (i/1.0))
            print("time.shape", time)

            print("\n === CELL 1  GraphRNN group by 2 Neighborhoods by features  ====") 
            with tf.variable_scope('decoder_1', reuse=tf.AUTO_REUSE) as scope:
            	global_state1 = graph_cell1( (f_xyz3, None, f_feat3, None, time), global_state1)
            	s_xyz1, s_color1, s_feat1, s_states1, time,nbrs1, cp1, extra  = global_state1
            	print("s_xyz1",s_xyz1)
            	print("s_feat1",s_feat1)
            	print("s_color1",s_color1)
            	print("s_states1",s_states1)
            	print("\n")
            
            print("\n === CELL 2 GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz2 =   s_xyz1        
            feat2 =  s_feat1
            states2 = s_states1
            time = tf.fill( (xyz2.shape[0],xyz2.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_2', reuse=tf.AUTO_REUSE) as scope:
                global_state2 = graph_cell2( (xyz2, None, feat2, states2, time), global_state2)
                s_xyz2, s_color2, s_feat2, s_states2, time,nbrs2, cp2,_ = global_state2
                print("s_xyz2",s_xyz2)
                print("s_feat2",s_feat2)
                print("s_color2",s_color2)
                print("s_states2",s_states2)
                print("\n")                

            print("\n === CELL 3  GraphRNN group by 2 Neighborhoods by features  ====") 
            xyz3 =   s_xyz2        
            feat3 =  s_feat2
            states3 = s_states2
            time = tf.fill( (xyz3.shape[0],xyz3.shape[1],1), (i/1.0))
            with tf.variable_scope('decoder_3', reuse=tf.AUTO_REUSE) as scope:
                global_state3 = graph_cell3( (xyz3, None, feat3, states3, time), global_state3)
                s_xyz3, s_color3,s_feat3, s_states3, time,nbrs3, cp3,_  = global_state3
                print("s_xyz3",s_xyz3)
                print("s_feat3",s_feat3)
                print("s_color3",s_color3)
                print("s_states3",s_states3)                
                print("\n")
         
            with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
                predicted_motion = tf.layers.conv1d(inputs=s_states3, filters=128, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc1')
                predicted_motion = tf.layers.conv1d(inputs=predicted_motion, filters=3, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc2')
                
         
            predicted_motions.append(predicted_motion)
            predicted_frame = xyz0 + predicted_motion      
            predicted_frames.append(predicted_frame)
                 
            #input is prediction
            input_frame = input_frame + predicted_motion
            
            self.extra.append(extra)
            self.out_s_xyz1.append(s_xyz1)
            self.out_s_color1.append(s_color1)
            self.out_s_feat1.append(s_feat1)
            self.out_s_states1.append(s_states1)
            self.out_s_xyz2.append(s_xyz2)
            self.out_s_color2.append(s_color2)
            self.out_s_feat2.append(s_feat2)
            self.out_s_states2.append(s_states2)
            self.out_s_xyz3.append(s_xyz3)
            self.out_s_color3.append(s_color3)
            self.out_s_feat3.append(s_feat3)
            self.out_s_states3.append(s_states3)
            self.out_nbrs1.append(nbrs1)
            self.out_nbrs2.append(nbrs2)
            self.out_nbrs3.append(nbrs3)
            self.out_cp1.append(cp1)
            self.out_cp2.append(cp2)
            self.out_cp3.append(cp3)            
            
        
        downsample_frames = frames                
        self.downsample_frames = downsample_frames
        self.predicted_motions = predicted_motions
        
        print("frames.shape", np.shape(frames))
        print("downsample_frames.shape", np.shape(downsample_frames) )
        print("predicted_frames.shape", np.shape(predicted_frames) )

        self.loss = self.emd = self.cd = 0
        self.emd_color = self.cd_color = 0
        self.diff =0
        self.frame_diff=[]
        self.frame_loss_cd=[]
        self.frame_loss_emd=[]

        for i in range(int(seq_length/2)):

            	# Select and split frames
            	print("downsample_frames[",i+int(seq_length/2),"] compare with -> predicted_frames[",i,"]")
            	frame = downsample_frames[i+int(seq_length/2)]
            	predicted_frame =predicted_frames[i]
            	frame_points = frame
            	predicted_frame_points = predicted_frame

            	# Geometry loss
            	match = tf_approxmatch.approx_match(frame_points,predicted_frame_points)
            	match_cost = tf_approxmatch.match_cost(frame_points, predicted_frame_points, match)
            	emd_distance = tf.reduce_mean(match_cost)
            	loss_emd = emd_distance
            	self.emd += loss_emd
            	self.frame_loss_emd.append(loss_emd)            	
            	
            	dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(predicted_frame_points, frame_points)
            	loss_cd = tf.reduce_mean(dists_forward+dists_backward)
            	self.cd += loss_cd
            	self.frame_loss_cd.append(loss_cd)    
            	
            	diff = abs(dists_forward) + abs(dists_backward)
            	self.frame_diff.append(diff)

            	
            	#Global Loss
            	self.loss += ( alpha*(loss_cd)  + (beta*loss_emd) )

        self.cd /= int(seq_length/2)  
        self.emd /= (int(seq_length/2)*num_points)
        #self.cd_color /= int(seq_length-1-context_frames)
        #self.emd_color /= (int(seq_length-1-context_frames)*(num_points))
        
        self.loss /= int(seq_length/2)

        if is_training == True :
        
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.predicted_motions = tf.stack(values=predicted_motions, axis=1)
        self.predicted_frames = tf.stack(values=predicted_frames, axis=1)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        if is_training == False:
        
        	print("Dont update paramenters")
        	params = tf.trainable_variables()
        	#print("params",params)


