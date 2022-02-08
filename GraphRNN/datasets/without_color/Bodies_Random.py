import os
import numpy as np

"""
Load all Human Bodies at random speed and start frame

"""
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
        folder = str(num_points)
        
        log_nr = 0
        print(" LOAD BODIES AT RANDOM DATASET ")

        if train:
            splits = [folder]
        else:
            splits = ['test']

        for split in splits:           
            print("split: ", split)
            split_path = os.path.join(root, split)
            split_path_color = os.path.join(root_color, split)

            #SELECT CHARACTER
            for charater in sorted (os.listdir(split_path)):
                charater_path = os.path.join(split_path, charater)
                charater_path_color = os.path.join(split_path_color, charater)
                
                if(charater != 'zBrian' ):
                
                  rnd_0 = np.random.randint(0,12)
                  rnd_01 = np.random.randint(0,12)
                  rnd_2 = np.random.randint(0,25)
                  rnd_1 = np.random.randint(0,25)
                  rnd_3 = np.random.randint(0,25)

                  for sequence in sorted(os.listdir(charater_path)):
                      if(sequence[0] != '0'): # Load all data

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
                      	
                      	
                      	log_nr= log_nr + 1

        print("[OK ] Dataset: ", np.shape(self.data) )
        #print("self.data_color", np.shape(self.data_color) )
        
               	                	                          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):

        nr_seq = len(self.data)	
        rand = np.random.randint(0,nr_seq)
        log_data = self.data[rand] 
        #log_data_color = self.data_color[rand]
                
        total_lenght = len(log_data)
        start_limit = total_lenght - self.seq_length 
        #print("start_limit:", start_limit)
        #print("total_lenght",total_lenght)
        
        # CHECK FOR ERROR
        if( start_limit < 0 or start_limit > 40  ):
        	error = 1
        	#get new sequence
        	while (error == 1):
        		#print("[ERROR] in [Seq] %d (of %d) lenght: %10d" %(rand, nr_seq, total_lenght ) )
        		#print("[ERROR] : start_limit :", start_limit)
        		#print("Ignore sequence")
        		#print("Get new sequence")
        		rand = np.random.randint(0,nr_seq)
        		log_data = self.data[rand]
        		total_lenght = len(log_data)
        		start_limit = total_lenght - self.seq_length 
        		if( start_limit > -1 and start_limit < 40):
        		  error = 0
        		else:
        		  error = 1

        if( start_limit == 0):
        	start =0
        else:
        	start = np.random.randint(0 ,start_limit)
        	           
        cloud_sequence = []
        
        for i in range(start, start+self.seq_length):
            pc = log_data[i]
            #pc_color = log_data_color[i]
            
            npoints = pc.shape[0]
            #sample_idx = np.random.choice(npoints, self.num_points, replace=False)
            
            cloud_sequence.append(pc)
            
        points= (np.stack(cloud_sequence, axis=0))
        #color= (np.stack(cloud_sequence_color, axis=0) )
       

        
        return ( points )
 

