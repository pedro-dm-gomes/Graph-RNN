import os
import sys
import pathlib
import glob
import numpy as np
import open3d as o3d
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",
                        help=f"Base directory of input.",
                        type=str)
    parser.add_argument("--output_dir",
                        help=f"Output directory [default=<same as input_dir>]",
                        type=str,
                        default="")
    return parser.parse_args()





args = parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

print("Load a ply point cloud and convert it")
print("Input_dir: ",input_dir)
print("Output_dir: ",output_dir)

pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

#Read all plys in folder
os.chdir(input_dir)

for ply_file in glob.glob("*.ply"):
  print("ply_file: ",ply_file)
  ply_path = os.path.join(input_dir, ply_file)
  
  #convert PLY --> numpy
  pcd = o3d.io.read_point_cloud(ply_path)
  
  base_filename = os.path.splitext(ply_file)[0]
  print("base_filename ",base_filename)
  npy_path = os.path.join(output_dir, base_filename + "." +'npy')
  
  pc_path= os.path.join(output_dir)
  np.save(npy_path, pcd.colors)



"""

python convert_ply_npy_points.py /media/pedro/HardDrive/Dataset/PLY_Bodys/800000/Simple/Moving_body_001z  --output_dir /media/pedro/HardDrive/Dataset/NPY_Bodys/800000/Simple/Moving_body_001z

"""

