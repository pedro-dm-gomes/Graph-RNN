import os
import sys
import numpy as np
import tensorflow as tf

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

class GraphRNNCell_2N_1MAX(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling

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
        inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        nbrs= None
        cp = None
        extra =None        

        return (P, C, F, S, T, nbrs, cp, extra)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1= inputs
        P2, C2, F2, S2, T2,_,_,_= states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        
        print("GraphRNN group by 2 Neighborhoods by geometry ")
        print("P1:",P1)
        print("P2:",P2)
        print("C1:",C1)
        print("C2:",C2)
        print("F1:",F1)
        print("F2:",F2)
        print("X1:",X1)
        print("S2:",S2)
        print("T1:",T1.shape)
        print("T2:",T2.shape)
        
                
        print("create adjacent matrix on feature space F1")
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        print("P1_adj_matrix",P1_adj_matrix)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample)
        print("P1_nn_idx",P1_nn_idx)
        
        # look at neighborhoodbood in P2
        print("create adjacent matrix on feature space F2 Fixed")
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F2, F1)
       	print("P2_adj_matrix",P2_adj_matrix)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k= nsample)
        print("P2_nn_idx",P2_nn_idx)
        
        if (knn == False) : # DO A BALL QUERY
        	print("\nBALL QUERY")
        	"""
        	idx, cnt = query_ball_point(radius, nsample, P1, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P1_nn_idx = tf.where(cnt > (nsample-1), idx, P1_nn_idx)
        	
        	idx, cnt = query_ball_point(radius, nsample, P2, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P2_nn_idx = tf.where(cnt > (nsample-1), idx, P2_nn_idx )
        	"""
        else:
        	print("KNN QUERY")

        
        # 2.1 Group P1 points
        P1_grouped = group_point(P1, P1_nn_idx)                      
        # 2.3 Group P color
        C1_grouped = group_point(C1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F1_grouped = group_point(F1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P time
        T1_grouped = group_point(T1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.2 Group P1 states
        if (X1 is not None):
        	S1_grouped = group_point(X1, P1_nn_idx)  
        
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        # 2.3 Group P color
        C2_grouped = group_point(C2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F2_grouped = group_point(F2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
        #2.4 Group S2 states
        S2_grouped = group_point(S2, P2_nn_idx)   
        # 2.4 Group P2 time
        T2_grouped = group_point(T2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels

                    
        
        #save point 100 neigbhood
        point_nr = 25
        print("P1_grouped[0][point_nr]",P1_grouped[0][point_nr])
        nbrs = tf.concat([P1_grouped[0][point_nr], P2_grouped[0][point_nr]], axis=0)     
        cp = P1[0][point_nr]
        
        ##  Neighborhood P1 "
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
        displacement_color = C1_grouped - C1_expanded           # batch_size, npoint, nsample, 3
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
        displacement_color_2 = C2_grouped - C1_expanded           # batch_size, npoint, nsample, 3
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
        	
        	correlation = tf.concat([X1_expanded, S1_grouped], axis=3)         
        	correlation = tf.concat([correlation, displacement ,displacement_feat, displacement_time], axis=3)
        	
        	correlation_2 = tf.concat([X1_expanded, S2_grouped], axis=3)
        	correlation_2 = tf.concat([correlation_2, displacement_2,displacement_feat_2, displacement_time_2], axis=3) 
        	             
        else:
        	F1_expanded = tf.tile(tf.expand_dims(F1, 2), [1, 1, nsample, 1])                        
        	correlation = tf.concat([displacement, displacement_feat, displacement_time], axis=3)
        	correlation_2 = tf.concat([displacement_2, displacement_feat_2,displacement_time_2], axis=3)         


        
        print("[1] correlation",correlation)
        print("[1] correlation_2",correlation_2)
        print("[1] correlation_1 = [S1_point | S1_neighborhodd | displacement | displacement_feat| displacement_time] " )
        
        #Unifty both correlations
        correlation = tf.concat([correlation, correlation_2], axis=2)
        print("[f] correlation",correlation)
        
        # 5. Fully-connected layer (the only parameters)
        with tf.variable_scope('graph-rnn') as sc:
        	S1 = tf.layers.conv2d(inputs=correlation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc')
        
        #S1_before_Max = S1
        
        # 6. Pooling
        if pooling=='max':
        	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)  
        	
        	    

        return (P1, C1, F1, S1, T1,  nbrs, cp ,displacement) 

""" ======================================="""
class GraphRNNCell_2N(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling

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
        inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        nbrs= None
        cp = None        

        return (P, C, F, S, T, nbrs, cp)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1= inputs
        P2, C2, F2, S2, T2,_,_= states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        
        print("GraphRNN group by 2 Neighborhoods by geometry ")
        print("P1:",P1)
        print("P2:",P2)
        print("C1:",C1)
        print("C2:",C2)
        print("F1:",F1)
        print("F2:",F2)
        print("X1:",X1)
        print("S2:",S2)
        print("T1:",T1.shape)
        print("T2:",T2.shape)
        
                
        print("create adjacent matrix on feature space F1")
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        print("P1_adj_matrix",P1_adj_matrix)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample)
        print("P1_nn_idx",P1_nn_idx)
        
        # look at neighborhoodbood in P2
        print("create adjacent matrix on feature space F2")
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F1, F2)
       	print("P2_adj_matrix",P2_adj_matrix)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k= nsample)
        print("P2_nn_idx",P2_nn_idx)
        
        if (knn == False) : # DO A BALL QUERY
        	print("\nBALL QUERY")
        	idx, cnt = query_ball_point(radius, nsample, P1, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P1_nn_idx = tf.where(cnt > (nsample-1), idx, P1_nn_idx)
        	
        	idx, cnt = query_ball_point(radius, nsample, P2, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	P2_nn_idx = tf.where(cnt > (nsample-1), idx, P2_nn_idx )

        else:
        	print("KNN QUERY")

        
        # 2.1 Group P1 points
        P1_grouped = group_point(P1, P1_nn_idx)                      
        # 2.3 Group P color
        C1_grouped = group_point(C1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F1_grouped = group_point(F1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P time
        T1_grouped = group_point(T1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        # 2.3 Group P color
        C2_grouped = group_point(C2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F2_grouped = group_point(F2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P time
        T2_grouped = group_point(T1, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels

        # 2.2 Group P1 states
        if (X1 is not None):
        	S1_grouped = group_point(X1, P1_nn_idx)                      
        else:
        	S1_grouped = P1_grouped
        if (S2 is not None):
        	S2_grouped = group_point(S2, P2_nn_idx)                        	
        else:
        	S2_grouped = P2_grouped	
        	
        print("S1_grouped",S1_grouped)	        	
        print("S2_grouped",S2_grouped)
        
        #save point 100 neigbhood
        point_nr = 600
        print("P1_grouped[0][point_nr]",P1_grouped[0][point_nr])
        nbrs = tf.concat([P1_grouped[0][point_nr], P2_grouped[0][point_nr]], axis=0)     
        cp = P1[0][point_nr]
        
        ##  Neighborhood P1 "
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
        displacement_color = C1_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F1_expanded = tf.expand_dims(F1, 2)                     # batch_size, npoint, 1,       3
        displacement_feat = F1_grouped - F1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate time displacements
        T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
        displacement_time = T1_grouped - T1_expanded           # batch_size, npoint, nsample, 3
              

        ##  Neighborhood P2 "
        # 3. Calculate displacements
        P2_expanded = tf.expand_dims(P2, 2)                     # batch_size, npoint, 1,       3
        displacement_2 = P1_grouped - P2_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        C2_expanded = tf.expand_dims(C2, 2)                     # batch_size, npoint, 1,       3
        displacement_color_2 = C2_grouped - C2_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F2_expanded = tf.expand_dims(F2, 2)                     # batch_size, npoint, 1,       3
        displacement_feat_2 = F2_grouped - F2_expanded           # batch_size, npoint, nsample, 3        
        #3.1 Calculate time displacements
        T2_expanded = tf.expand_dims(T2, 2)                     # batch_size, npoint, 1,       3
        displacement_time2 = T2_grouped - T2_expanded           # batch_size, npoint, nsample, 3

        # 4. Concatenate X1, S2 and displacement
        if X1 is not None:
        	X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])                
        	correlation = tf.concat([S1_grouped, X1_expanded], axis=3)         
        	correlation = tf.concat([correlation, displacement, displacement_color, displacement_feat,displacement_time], axis=3)
        	               
        else:
        	correlation = tf.concat([S1_grouped, displacement, displacement_color, displacement_feat,displacement_time], axis=3)
        
          
        X2_expanded = tf.tile(tf.expand_dims(S2, 2), [1, 1, nsample, 1])
        correlation_2 = tf.concat([S2_grouped, X2_expanded], axis=3)
        correlation_2 = tf.concat([correlation_2, displacement_2, displacement_color_2, displacement_feat_2,displacement_time2], axis=3)
        
        print("[1] correlation",correlation)
        print("[1] correlation_2",correlation_2)
        print("[1] correlation_1 = [S1_point | S1_neighborhodd | displacement | displacement_color | displacement_feat | displacement_time] " )
        
       
        # 5. Fully-connected layer (the only parameters) 
        with tf.variable_scope('graph-rnn_neighborhood', reuse=tf.AUTO_REUSE ) as sc:
        	N1 = tf.layers.conv2d(inputs=correlation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='graph-rnn_neighborhood_1')
        
        	N2 = tf.layers.conv2d(inputs=correlation_2, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='graph-rnn_neighborhood_2')

        print("N1", N1)
        print("N2", N2)
        
        # 6. Pooling
        if pooling=='max':
        	N1 = tf.reduce_max(N1, axis=[2], keepdims=False)
        	N2 = tf.reduce_max(N2, axis=[2], keepdims=False)
        elif pooling=='avg':
        	N1 =tf.reduce_mean(N1, axis=[2], keepdims=False)    
        	N2 =tf.reduce_mean(N2, axis=[2], keepdims=False)         
       
        print("N1", N1)
        print("N2", N2)
        final_correlation = tf.concat([N1, N2], axis=2)
        
        print("final_correlation",final_correlation)
        with tf.variable_scope('graph-rnn_final', reuse=tf.AUTO_REUSE) as scope:
                S1 = tf.layers.conv1d(inputs=final_correlation, filters=self.out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='graph-rnnfinal')
                        
        print("S1 ", S1)
        	      
        return (P1, C1, F1, S1, T1,  nbrs, cp)

""" *******************************************************************************************   """

class GraphRNNCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling

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
        inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        nbrs= None
        cp = None        
        extra =None        

        return (P, C, F, S, T, nbrs, cp, extra)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1= inputs
        P2, C2, F2, S2, T2,_,_,_= states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        
        print("GraphRNN Grouping by Features with 2 times")
        print("P1:",P1)
        print("P2:",P2)
        print("C1:",C1)
        print("C2:",C2)
        print("F1:",F1)
        print("F2:",F2)
        print("X1:",X1)
        print("S2:",S2)
        print("T1:",T1.shape)
        print("T2:",T2.shape)
        
        #create Full point cloud
        P = tf.concat([P1, P2], axis=1)
        C = tf.concat([C1, C2], axis=1)
        F = tf.concat([F1, F2], axis=1)
        T = tf.concat([T1, T2], axis=1)
        
        print("P",  P)
        
        print("create big adjacent matrix on feature space")
        big_adj_matrix = tf_util.pairwise_distance(F)
        print("big_adj_matrix",big_adj_matrix)
        big_nn_idx = tf_util.knn(big_adj_matrix, k= nsample)
        print("big_nn_idx",big_nn_idx)

        #only get the frist half
        idx_knn,_ = tf.split(big_nn_idx, 2, axis = 1)
        print("idx_knn",idx_knn)
                
        if knn == True:
        	print("KNN QUERY")
        	idx = idx_knn
        else:
        	print("BALL QUERY")
        	idx_ball, cnt = query_ball_point(radius, nsample, P, P1)
        	print("cnt",cnt)
        	print("idx_ball",idx_ball)
        	#_, idx_knn = knn_point(nsample, P, P1)
        	cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        	print("cnt",cnt)
        	idx = tf.where(cnt > (nsample-1), idx_ball, idx_knn)
        	print("idx",idx)
        	
        	
      
        # 2.1 Group P points
        P2_grouped = group_point(P, idx)                      
        # 2.3 Group P color
        C2_grouped = group_point(C, idx)                       # batch_size, npoint, nsample, out_channels# 2.4 Group P feat
        F2_grouped = group_point(F, idx)                       # batch_size, npoint, nsample, out_channels
        # 2.4 Group P time
        T2_grouped = group_point(T, idx)                       # batch_size, npoint, nsample, out_channels
        
        # 2.2 Group P2 states
        if (X1 is not None):
        	S = tf.concat([X1, S2], axis=1)
        	print("S",S)  
        	S2_grouped = group_point(S, idx)                       # batch_size, npoint, nsample, out_channels
        	print("S2_grouped",S2_grouped)
        else:
        	S2_grouped = P2_grouped
        	
        #save point 100 neigbhood
        nbrs = P2_grouped[0][2]
        cp =P1[0][2]
        
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        #3.1 Calculate color displacements
        C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
        displacement_color = C2_grouped - C1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate time displacements
        T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
        displacement_time = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3
        #3.1 Calculate feature displacements
        F1_expanded = tf.expand_dims(F1, 2)                     # batch_size, npoint, 1,       3
        displacement_feat = F2_grouped - F1_expanded           # batch_size, npoint, nsample, 3
        
        
        # 4. Concatenate X1, S2 and displacement
        if X1 is not None:
        	X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])                
        	correlation = tf.concat([S2_grouped, X1_expanded], axis=3)
        	correlation = tf.concat([correlation, displacement, displacement_time], axis=3)
        else:
        	correlation = tf.concat([S2_grouped, displacement, displacement_time], axis=3)

        print("[1] correlation",correlation)
        print("[1] correlation = S2 |S1 | displacement | displacement_color | displacement_time" )
        
        # 5. Fully-connected layer (the only parameters)
        with tf.variable_scope('graph-rnn') as sc:
        	S1 = tf.layers.conv2d(inputs=correlation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc')
        
        # 6. Pooling
        if pooling=='max':
        	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)  
        	      
        return (P1, C1, F1, S1, T1,  nbrs, cp , cp)

""" *******************************************************************************************   """


def point_rnn(P1,
              P2,
              C1,
              C2,
              F1,
              F2,
              X1,
              S2,
              T1,
              T2,
              prev_idx,
              radius,
              nsample,
              out_channels,
              knn=False,
              pooling='max',
              scope='point_rnn'):
              
    """
    Input:
        P1:     (batch_size, npoint, 3)
        P2:     (batch_size, npoint, 3)
        X1:     (batch_size, npoint, feat_channels) PointStates
        S2:     (batch_size, npoint, out_channels)
    Output:
        S1:     (batch_size, npoint, out_channels)
    """
    
    print("point-rrn color feat")
    print("P1:",P1)
    print("P2:",P2)
    print("C1:",C1)
    print("C2:",C2)
    print("F1:",F1)
    print("F2:",F2)
    print("X1:",X1)
    print("S2:",S2)
    print("T1:",T1.shape)
    print("T2:",T2.shape)
    
    
    #create Full point cloud
    P = tf.concat([P1, P2], axis=1)
    C = tf.concat([C1, C2], axis=1)
    F = tf.concat([F1, F2], axis=1)
    T = tf.concat([T1, T2], axis=1)
    print("P",P)
    print("C",C)
    print("F",F)              
    print("T",T)

    # 1. Sample points
    if knn:
        print("knn search")
        _, idx = knn_point(nsample, P, P1)
        print("idx", idx)
    else:
    	print("use old knn search")
    	idx = prev_idx
    	print("idx", idx)


    # 2.1 Group P points
    P2_grouped = group_point(P, idx)                       # batch_size, npoint, nsample, 3
    # 2.3 Group P color
    C2_grouped = group_point(C, idx)                       # batch_size, npoint, nsample, out_channels
    # 2.4 Group P feat
    F2_grouped = group_point(F, idx)                       # batch_size, npoint, nsample, out_channels
    # 2.4 Group P time
    T2_grouped = group_point(T, idx)                       # batch_size, npoint, nsample, out_channels
    
    #print("T2_grouped",T2_grouped)
    
    # 2.2 Group P2 states
    if (X1 is not None):
    	print("group  features")
    	S = tf.concat([X1, S2], axis=1)
    	print("S",S)  
    	S2_grouped = group_point(S, idx)                       # batch_size, npoint, nsample, out_channels
    else:
    	S2_grouped = P2_grouped
    	
    	
    #save point 100 neigbhood
    nbrs = P2_grouped[0][100]
    cp =P1[0][100]
    


    # 3. Calculate displacements
    P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
    displacement = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
    #3.1 Calculate color displacements
    C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
    displacement_color = C2_grouped - C1_expanded           # batch_size, npoint, nsample, 3
    #3.1 Calculate time displacements
    T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
    displacement_time = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3
    
     #3.1 Calculate feature displacements
    F1_expanded = tf.expand_dims(F1, 2)                     # batch_size, npoint, 1,       3
    displacement_feat = F2_grouped - F1_expanded           # batch_size, npoint, nsample, 3

    # 4. Concatenate X1, S2 and displacement
    if X1 is not None:
        X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])                # batch_size, npoint, sample,  feat_channels
        correlation = tf.concat([S2_grouped, X1_expanded], axis=3)                      # batch_size, npoint, nsample, feat_channels+out_channels
        correlation = tf.concat([correlation, displacement, displacement_color, displacement_feat, displacement_time], axis=3)                    # batch_size, npoint, nsample, feat_channels+out_channels+3
    else:
        correlation = tf.concat([S2_grouped, displacement, displacement_color, displacement_feat, displacement_time], axis=3)                     # batch_size, npoint, nsample, out_channels+3

    print("[1] correlation",correlation)
    print("[1] correlation = S2 |S1 | displacement | displacement_color | displacement_feat| displacement_time" )
    

    # 5. Fully-connected layer (the only parameters)
    with tf.variable_scope(scope) as sc:
        S1 = tf.layers.conv2d(inputs=correlation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc')

    # 6. Pooling
    if pooling=='max':
    	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
    	return(S1, nbrs, cp, idx)
    elif pooling=='avg':
    	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)        
    	return (S1 , nbrs, cp, idx)
    

class PointRNNCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling

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
        P, C, F, X, T, idx = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]
        inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        nbrs= None
        cp = None
        

        return (P, C, F, S, T, nbrs, cp, idx)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1, idx = inputs
        P2, C2, F2, S2, T2, _ ,_ ,_= states
    
        S1, nbrs, cp, idx = point_rnn(P1, P2, C1, C2, F1, F2, X1, S2, T1, T2, idx, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)

        return (P1, C1, F1, S1, T1,  nbrs, cp, idx)


""" *******************************************************************************************   """
        
def point_feat(P1,
              C1,
              F1,
              radius,
              nsample,
              out_channels,
              knn=False,
              pooling='max',
              scope='point_rnn'):
    """
    Input:
        P1:     (batch_size, npoint, 3)
        C1:     (batch_size, npoint, feat_channels)
    Output:
        F1:     (batch_size, npoint, out_channels)
    """
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
    print("P1_grouped",P1_grouped)
    
    # 3. Calcaulate displacements
    P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
    displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
    C1_expanded = tf.expand_dims(C1, 2)                     # batch_size, npoint, 1,       3
    displacement_color = C1_grouped - C1_expanded                 # batch_size, npoint, nsample, 3   


    # 4. Concatenate X1, S2 and displacement
    if F1 is not None:
    	F1_grouped = group_point(F1, idx)     
    	correlation =  tf.concat([P1_grouped, F1_grouped], axis=3) 
    	correlation = tf.concat([correlation,displacement, displacement_color], axis=3)
    else:
    	correlation = tf.concat([P1_grouped, displacement, displacement_color], axis=3)                     # batch_size, npoint, nsample, out_channels+3
    
    print("[2] correlation",correlation)
    
    # 5. Fully-connected layer (the only parameters)
    with tf.variable_scope(scope) as sc:
        F1 = tf.layers.conv2d(inputs=correlation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc')

    # 6. Pooling
    if pooling=='max':
        return tf.reduce_max(F1, axis=[2], keepdims=False)
    elif pooling=='avg':
        return tf.reduce_mean(F1, axis=[2], keepdims=False)    
      
      

        
class PointFeatureCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling


    def __call__(self, inputs):

        P1, C1, F1, S1 = inputs
        
        F1 = point_feat(P1, C1, F1, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        

        return (P1, C1, F1, S1)


