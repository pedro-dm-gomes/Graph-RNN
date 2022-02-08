"""
This file defines the Graph modules that use Color
"""
import os
import sys
import numpy as np
import tensorflow as tf
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))


from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point, knn_feat
from tf_interpolate import three_nn, three_interpolate
import tf_util

"""
List of Modules:
	1.GraphRNNCell
	2.GraphFeatureCell         	  	
"""

""" ================================================== 
               GRAPH-RNN                         
 ================================================== 
Module info: Standard Graph-RNN convolution with color
"""

class GraphRNNCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max',
                 activation= None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        self.activation = activation

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, C, F, X, T = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]
        #inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        #C = None
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        extra = None        

        return (P, C, F, S, T, extra)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1= inputs
        P2, C2, F2, S2, T2, extra = states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        activation = self.activation
    
        # Create adjacent matrix on feature space F1
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample)

        # look at neighborhoodbood in P2
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F2, F1)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k= nsample)

        
        if (knn == False) : # DO A BALL QUERY
        	print("\nBALL QUERY NOT IMPLEMENTED")
        
        # 2.1 Group P1 points
        P1_grouped = group_point(P1, P1_nn_idx)                      
        # 2.3 Group P color
        #C1_grouped = group_point(C1, P1_nn_idx)  
        # 2.4 Group P feat
        F1_grouped = group_point(F1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P time
        T1_grouped = group_point(T1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.2 Group P1 states
        if (X1 is not None):
        	S1_grouped = group_point(X1, P1_nn_idx)  
        
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        # 2.3 Group P color
        #C2_grouped = group_point(C2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F2_grouped = group_point(F2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
        #2.4 Group S2 states
        S2_grouped = group_point(S2, P2_nn_idx)   
        # 2.4 Group P2 time
        T2_grouped = group_point(T2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
 
        ##  Neighborhood P1"
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        #C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
        #displacement_color = C1_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F1_expanded = tf.expand_dims(F1, 2)                     # batch_size, npoint, 1,       3
        displacement_feat = F1_grouped - F1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate time displacements
        T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
        displacement_time = T1_grouped - T1_expanded           # batch_size, npoint, nsample, 3
              
        ##  Neighborhood P2 "
        # 3. Calculate displacements
        P2_expanded = tf.expand_dims(P2, 2)                     # batch_size, npoint, 1,       3
        displacement_2 = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        C2_expanded = tf.expand_dims(C2, 2)                     # batch_size, npoint, 1,       3
        #displacement_color_2 = C2_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F2_expanded = tf.expand_dims(F2, 2)                     # batch_size, npoint, 1,       3
        displacement_feat_2 = F2_grouped - F1_expanded           # batch_size, npoint, nsample, 3        
        #3.1 Calculate time displacements
        T2_expanded = tf.expand_dims(T2, 2)                     # batch_size, npoint, 1,       3
        displacement_time_2 = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3

        # 4. Concatenate X1, S2 and displacement
        if X1 is not None:
        	X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])  
        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                
        	
        	concatenation = tf.concat([X1_expanded, S1_grouped], axis=3)         
        	concatenation = tf.concat([concatenation, displacement ,displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([X1_expanded, S2_grouped], axis=3)
        	concatenation_2 = tf.concat([concatenation_2, displacement_2,displacement_feat_2, displacement_time_2], axis=3) 
        	             
        else:

        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                        
        	concatenation = tf.concat([displacement, displacement_feat, displacement_time], axis=3)
        	concatenation_2 = tf.concat([displacement_2, displacement_feat_2,displacement_time_2], axis=3)         

        #Unifty both concatenations
        concatenation = tf.concat([concatenation, concatenation_2], axis=2)

        
        # 5. Fully-connected layer (the only parameters)
        with tf.variable_scope('graph-rnn') as sc:
        	S1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='fc')
        
        #S1_before_Max = S1
        
        # 6. Pooling
        if pooling=='max':
        	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)  
        	
        return (P1, C1, F1, S1, T1, extra) 


""" ================================================== """
"""               GRAPH-FEATURE CELL                  """
""" ================================================== """

        
def graph_feat(P1,
              C1,
              F1,
              radius,
              nsample,
              out_channels,
              activation,
              knn=True,
              pooling='max',
              scope='graph_feat'):


   
    # 1. Sample points
    if knn:
    	_, idx = knn_point(nsample, P1, P1)
    else:
        idx, cnt = query_ball_point(radius, nsample, P1, P1)
        _, idx_knn = knn_point(nsample, P1, P1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)

    # 2.1 Group P2 points
    P1_grouped = group_point(P1, idx)                       # batch_size, npoint, nsample, 3
    C1_grouped = group_point(C1, idx)
    
    # 3. Calcaulate displacements
    P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
    displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
    C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
    displacement_color = C1_grouped - C1_expanded                 # batch_size, npoint, nsample, 3   


    # 4. Concatenate X1, S2 and displacement
    if F1 is not None:
    	F1_grouped = group_point(F1, idx)     
    	concatenation =  tf.concat([P1_grouped, F1_grouped], axis=3) 
    	concatenation = tf.concat([concatenation,displacement], axis=3)
    	concatenation = tf.concat([concatenation,displacement_color], axis=3)
    else:
    	concatenation = tf.concat([P1_grouped, displacement], axis=3)
    	concatenation = tf.concat([concatenation,displacement_color], axis=3)
    
    #print("Concatenation",concatenation)
    
    # 5. Fully-connected layer (the only parameters)
    with tf.variable_scope(scope) as sc:
        F1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='fc')

    # 6. Pooling

    if pooling=='max':
        F1= tf.reduce_max(F1, axis=[2], keepdims=False)
    elif pooling=='avg':
        F1= tf.reduce_mean(F1, axis=[2], keepdims=False)    

    return (F1)
            
class GraphFeatureCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max',
                 activation = None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        self.activation = activation


    def __call__(self, inputs):

        P1, C1, F1, S1 = inputs
        
        F1 = graph_feat(P1, C1, F1, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, activation = self.activation)
        
        
        return (P1, C1, F1, S1)
        

	         	             
