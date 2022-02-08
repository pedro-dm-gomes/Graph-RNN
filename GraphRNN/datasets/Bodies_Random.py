import os
import numpy as np

class Bodys(object):
    def __init__(self, root='/home/pedro/Desktop/Datasets/NPYs', seq_length=12, num_points=4000, train=True):
        
        root_color = root + '_Color'
              
        
        rnd_person_1 = np.random.randint(0,15) #-1
        rnd_person_2 = np.random.randint(0,15)
        rnd_person_3 = np.random.randint(0,15)
        rnd_person_4 = np.random.randint(0,15)
        rnd_person_5 = np.random.randint(0,15)

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []
        self.data_color = []
        
        
        print("seq_length",seq_length)
        print("num_points",num_points)
        points_folder =str(num_points)
        print("points_folder: ",points_folder)
        
        log_nr = 0
        print("    FULL RANDOM DATASET ")

        if train:
            splits = [points_folder]
        else:
            splits = ['test']

        for split in splits:           
            #print("split: ", split)
            split_path = os.path.join(root, split)
            split_path_color = os.path.join(root_color, split)
            #print("Point path:",split_path)
            #print("Color path", split_path_color)

            #SELECT CHARACTER
            for charater in sorted (os.listdir(split_path)):
                charater_path = os.path.join(split_path, charater)
                charater_path_color = os.path.join(split_path_color, charater)
                

                if(charater != 'ALL' ):
                
                  rnd_0 = np.random.randint(0,12)
                  rnd_01 = np.random.randint(0,12)
                  rnd_2 = np.random.randint(0,25)
                  rnd_1 = np.random.randint(0,25)
                  rnd_3 = np.random.randint(0,25)
                  
                  #print("[ ", charater ," ] Load: ",letters[rnd_1],letters[rnd_2],always_letters[rnd_0], always_letters[rnd_01])
                  for sequence in sorted(os.listdir(charater_path)):
                      if(sequence[0] != '0'): #Load All dataset

                      	fps = np.random.randint(1,4)
                      	odd = np.random.randint(0,2)
                      	sequence_path = os.path.join(charater_path, sequence)
                      	sequence_path_color = os.path.join(charater_path_color, sequence)
                      	#print("[%10d] [%s] (1/%d fps)"%(log_nr,sequence, fps) )
                      	
                      	# LOAD POINTS
                      	log_data = []
                      	frame = odd
                      	for npy in sorted(os.listdir(sequence_path)):
                      		#Load at diferent speeds
                      		if( frame %(fps) == 0):
                      		  npy_file = os.path.join(sequence_path, npy)
                      		  npy_data = np.load(npy_file)
                      		  log_data.append(npy_data)
                      		frame = frame +1                       		                         		  	
                      	self.data.append(log_data)   
                      	
                      	
                      	# LOAD COLOR
                      	log_data_color = []
                      	frame = odd
                      	for npy in sorted(os.listdir(sequence_path_color)):
                      		#Load at diferent speeds
                      		if( frame %(fps) == 0):
                      		  npy_file = os.path.join(sequence_path_color, npy)
                      		  npy_data = np.load(npy_file)
                      		  #print("color: ",npy_data)
                      		  log_data_color.append(npy_data)
                      		frame = frame +1                       		                         		  	
                      	self.data_color.append(log_data_color)
                      	
                      	
                      	log_nr= log_nr + 1

        print("self.data", np.shape(self.data) )
        print("self.data_color", np.shape(self.data_color) )
        
               	                	                          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):

        nr_seq = len(self.data)	
        rand = np.random.randint(0,nr_seq)
        log_data = self.data[rand] 
        log_data_color = self.data_color[rand]
                
        total_lenght = len(log_data)
        total_lenght_color = len(log_data_color)        
        start_limit = total_lenght - self.seq_length 
        start_limit_color = total_lenght_color - self.seq_length 
        
        # CHECK FOR ERROR
        error = error_1 = error_2 = error_3 =0
        if( start_limit < 0 or start_limit > 40  ):
        	error_1 = 1
        	print("[DATA LOADING GEO ERROR]")
        if( start_limit_color < 3 or start_limit_color > 40  ):
        	error_2 = 1
        	print("[DATA LOADING COLOR ERROR]")	
        if( total_lenght != total_lenght_color ):
        	error_3 = 1
        	print("[DATA LOADING MISSMATCH ERROR]")

        #get new sequence
        if( error_1==1 or error_2 ==1 or error_3 ==1):
        	error=1
        	while (error == 1):
        		#print("[ERROR] in [Seq] %d (of %d) lenght: %10d" %(rand, nr_seq, total_lenght ) )
        		
        		print("Get new sequence")
        		rand = np.random.randint(0,nr_seq)
        		log_data = self.data[rand]
        		log_data_color = self.data_color[rand]
        		#print("log_data", np.shape(log_data) )
        		#print("log_data_color", np.shape(log_data_color) )
        
        		total_lenght = len(log_data)
        		total_lenght_color = len(log_data_color)        
        		start_limit = total_lenght - self.seq_length 
        		start_limit_color = total_lenght_color - self.seq_length 
        		
        		error = error_1 = error_2 = error_3 =0
        		if( start_limit < 0 or start_limit > 40  ):
        			error_1 = 1
        			#print("[GEO ERROR]")
        		if( start_limit_color < 0 or start_limit_color > 40  ):
        			error_2 = 1
        			#print("[COLOR ERROR]")
        		if( total_lenght != total_lenght_color ):
        			error_3 = 1
        			#print("[MISSMATCH ERROR]")
        		
        		if( error_1==1 or error_2 ==1 or error_3 ==1):
        			error=1
        if( start_limit == 0):
        	start =0
        else:
        	start = np.random.randint(0 ,start_limit)    
        #print("[GO] [Seq] %d (of %d) start %d (of %d)" %(rand, nr_seq, start, total_lenght ) )

   
        cloud_sequence = []
        cloud_sequence_color = []
        
        for i in range(start, start+self.seq_length):
            pc = log_data[i]
            pc_color = log_data_color[i]
            
            npoints = pc.shape[0]

            #sample_idx = np.random.choice(npoints, self.num_points, replace=False)
            
            cloud_sequence.append(pc)
            cloud_sequence_color.append(pc_color)
            


        points= (np.stack(cloud_sequence, axis=0))
        color= (np.stack(cloud_sequence_color, axis=0) )
       

        
        return ( points, color )
 

