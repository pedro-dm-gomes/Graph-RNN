# Dataset Creation

To create the Sythenthic Bodies Dataset we followed the work form Irene et al from "Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks" 
In case of any doubt you will find a diferent code to achive the similar data creation at https://github.com/jelmr/pc_temporal_interpolation

1. Download the FBX files from Mixamo.

2. We recomentd to organize the FBXs files with the following tree structure
```
data
|-FBX
  |-Astra
    |- Arial_Evade.fbx
    |- Butterfly_Twirl.fbx
    |- (...)
  |-Brian
    |- Chicken_Dance.fbx
    |- Hit_to_Body.fbx
  (...)
```

3. To convert the Fbx file to point cloud run the following python:
You will need to edit the script with the coorect paths to the directories in your computer.

    `python create_dataset_color_full_body.py`

The python sctrip will  convert .FBX -> OBJ -> PLY. The final step the PLY are converted to NPY. For each point cloud there will a npy file for the points and npy file for the color.

After the script you should have:
```
data
|-FBX
  |-Astra
    |- Arial_Evade.fbx
|-OBJ_Bodys
  |-Astra
    |- Arial_Evade
      |- frame_000001.mtl
      |- frame_000001.obj
|-PLY_Bodys
  |-800000
    |-Astra
      |- Arial_Evade
        |- frame_000001.ply
        |- frame_000002.ply
|-NPY_Bodys
  |-800000
    |-Astra
      |- Arial_Evade
        |- frame_000001.npy
        |- frame_000002.npy
|-NPY_Bodys_Color
  |-800000
    |-Astra
      |- Arial_Evade
        |- frame_000001.npy
        |- frame_000002.npy
  ```


4. Downsample the point clouds using farthest point sampling algoritm from 800,000 points to 4,000 points. 
You will need to compile the code in GraphRNN folder for this step.

    `python /GraphRNN/Downsample_all_sequences.py`

5. Set the correct path in GraphRNN files.

For example in `train-GraphRNN_ShortTerm_color.py` you will need to change the following line  for you personal directory with the npy files.
parser.add_argument('--data-dir', default='/home/uceepdg/profile.V6/Desktop/Datasets/NPYs_Bodys')
