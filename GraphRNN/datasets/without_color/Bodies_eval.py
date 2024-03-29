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
        
        
        letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T','U','V','W','X','Y','Z'] #25
        always_letters = ['B','C','D','F','G','H','I','J','P','R','T','W'] #12
        persons =['Megan','Sophie','Brian','Douglas','Joe','Kate','Lewis','Astra','Louise','Malcom', 'Martha','Remmy', 'Regina', 'Roth', 'Stefani'] #15 sequencias
        
        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []
        self.data_color = []
        
        
        print("seq_length",seq_length)
        print("num_points",num_points)
        
        log_nr = 0
        print("    FULL RANDOM DATASET ")
        nr_pts=str(num_points)

        if train:
            splits = ['4000']
        else:
            splits = ['test/'+nr_pts]

        for split in splits:           
            print("split: ", split)
            split_path = os.path.join(root, split)
            split_path_color = os.path.join(root_color, split)
            print("split_path",split_path)

            #SELECT CHARACTER
            for charater in sorted (os.listdir(split_path)):
                charater_path = os.path.join(split_path, charater)
                charater_path_color = os.path.join(split_path_color, charater)
                

                if(charater != 'JPEG' ):
                  print("[ ", charater ," ]")
                  for sequence in sorted(os.listdir(charater_path)):
                      if( 0 == 0 ):

                      	sequence_path = os.path.join(charater_path, sequence)
                      	sequence_path_color = os.path.join(charater_path_color, sequence)

                      	# LOAD POINTS
                      	log_data = []
                      	frame =0
                      	fps=1
                      	print("[%10d] [%s] (1/%d fps)"%(log_nr,sequence, fps) )
                      	for npy in sorted(os.listdir(sequence_path)):
                      		#Load at diferent speeds
                      		if( frame %(fps) == 0):
                      		  npy_file = os.path.join(sequence_path, npy)
                      		  npy_data = np.load(npy_file)
                      		  log_data.append(npy_data)
                      		frame = frame +1                       		                         		  	
                      	self.data.append(log_data)   
                      	log_nr= log_nr + 1
                      	
                      	log_data = []
                      	frame = 0
                      	fps = 2
                      	print("[%10d] [%s] (1/%d fps)"%(log_nr,sequence, fps) )
                      	for npy in sorted(os.listdir(sequence_path)):
                      		#Load at diferent speeds
                      		if( frame %(fps) == 0):
                      		  npy_file = os.path.join(sequence_path, npy)
                      		  npy_data = np.load(npy_file)
                      		  log_data.append(npy_data)
                      		frame = frame +1                       		                         		  	
                      	self.data.append(log_data)
                      	log_nr= log_nr + 1 

                      	log_data = []
                      	frame = 0
                      	fps= 3
                      	print("[%10d] [%s] (1/%d fps)"%(log_nr,sequence, fps) ) 
                      	for npy in sorted(os.listdir(sequence_path)):
                      		#Load at diferent speeds
                      		if( frame %(fps) == 0):
                      		  npy_file = os.path.join(sequence_path, npy)
                      		  npy_data = np.load(npy_file)
                      		  log_data.append(npy_data)
                      		frame = frame +1                       		                         		  	
                      	self.data.append(log_data) 
                      	log_nr= log_nr + 1

        print("self.data", np.shape(self.data) )
        #print("self.data_color", np.shape(self.data_color) )
        
               	                	                          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, nr):

        rand =nr
        log_data = self.data[rand] 
        #log_data_color = self.data_color[rand]
                
        total_lenght = len(log_data)
        start_limit = total_lenght - self.seq_length 
        start = 0
				
        #print("[GO] [Seq] %d (of %d) start %d (of %d)" %(rand, 152, start, total_lenght ) )

   
        cloud_sequence = []
        #cloud_sequence_color = []
        
        for i in range(start, start+self.seq_length):
            pc = log_data[i]
            #pc_color = log_data_color[i]
            
            npoints = pc.shape[0]
            #sample_idx = np.random.choice(npoints, self.num_points, replace=False)
            
            cloud_sequence.append(pc)
            #cloud_sequence_color.append(pc_color)
            
        points= (np.stack(cloud_sequence, axis=0))
        #color= (np.stack(cloud_sequence_color, axis=0) )
       

        
        return ( points )


