
"""
This script will create Point cloud with 800,000 points  from FBX file
"""

import os
import subprocess
import os.path as osp
import glob
import pathlib
import time

processes = set()

""" You have to setup your own paths """
""" You nedd a FBX Folder containing fbx files from mixamo """

python_path="fbx_to_obj_object.py"
root= '/media/pedro/HardDrive/Dataset/FBX'
base_obj= '/media/pedro/HardDrive/Dataset/OBJ_Bodys'
base_ply= '/media/pedro/HardDrive/Dataset/PLY_Bodys'
base_npy= '/media/pedro/HardDrive/Dataset/NPY_Bodys'
base_npy_color = '/media/pedro/HardDrive/Dataset/NPY_Bodys_Color'


for character in os.listdir(root):
	print("character: ",character)
	character_path = os.path.join(root, character)
	if (character != 'ALL'):
		for sequence in os.listdir(character_path):
			if(sequence != 'Running.fbx ):
				print("sequence: ",sequence)
				
				print("\nConvert FBX to OBJs")
				input_fbx = os.path.join(character_path,sequence)
				print("input_fbx: ",input_fbx)
				obj_output = os.path.join(base_obj,character )
				obj_output = os.path.join(obj_output,sequence)
				obj_output=os.path.splitext(obj_output)[0]
				print("obj_output: ",obj_output)
				
				#Call Blender Convert to OBJ
				cmd ='blender --background -P ' + 'fbx_to_obj_body.py' +' '+ input_fbx +' '+ obj_output
				#print("cmd :",cmd)		
				subprocess.call([ cmd], shell=True)
			
		
				print("\nConvert OBJ to PLY")
				nr_points= '800000'
				path ='800000'
				path='test'
				
				ply_output = os.path.join(base_ply,path)
				ply_output = os.path.join(ply_output,character)
				ply_output = os.path.join(ply_output,sequence)
				ply_output=os.path.splitext(ply_output)[0]
				cmd ='python obj_to_ply_sample.py ' + obj_output + ' --output_dir ' +  ply_output +' --n ' + nr_points
				#print("cmd :",cmd)
				subprocess.call([ cmd], shell=True)
					
				
			
				print("\nConvert to NPY Points & Color ")
				npy_output =os.path.join(base_npy,path)
				npy_output =os.path.join(npy_output,character)
				npy_output =os.path.join(npy_output,sequence)
				npy_output =os.path.splitext(npy_output)[0]
				print("npy_output: ",npy_output)
				cmd =' python convert_ply_npy_points.py  ' + ply_output + ' --output_dir ' + npy_output
				#print("cmd :",cmd)
				subprocess.call([ cmd], shell=True)
				
				npy_output =os.path.join(base_npy_color,path)
				npy_output =os.path.join(npy_output,character)
				npy_output =os.path.join(npy_output,sequence)
				npy_output =os.path.splitext(npy_output)[0]
				print("npy_output: ",npy_output)
				cmd =' python convert_ply_npy_color.py  ' + ply_output + ' --output_dir ' + npy_output
				#print("cmd :",cmd)
				subprocess.call([ cmd], shell=True)
				
		
		print("\n")



"""

blender --background -P fbx_to_obj_object.py /media/pedro/SSD_DISK/FBXs/Louise/Jump.fbx /media/pedro/SSD_DISK/OBJS_SHOES/Louise/Jump.fbx

python obj_to_ply_sample.py  /home/pedro/Desktop/OBJs/Stefani/Walk_In_Circle_Shoes --output_dir /home/pedro/Desktop/OBJs/Stefani/Walk_In_Circle_Shoes_sample_10_000pts --n 10000  



python convert_ply_npy.py /home/pedro/Desktop/OBJs/Stefani/Walk_In_Circle_Shoes_sample_10_000pts  --output_dir /home/pedro/Desktop/NPYs/train/Stefani/Walk_In_Circle_Shoes_sample_10_000pts 



"""



