# Dataset Creation

To create the Sythenthic Bodies Dataset we followed the work form Irene et al from "Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks" 
In case of any doubt you will find a diferent code to achive the same data creation at https://github.com/jelmr/pc_temporal_interpolation

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

3. Run the following python code to convert the FBX animation to-> OBJ -> PLY -> NPY

    python create_dataset_color_full_body.py

4. Downsample the Sequences using Farthest point sampling algoritm from 800,000 points to 4,000 points

    python create_dataset_color_full_body.py

5. Set the correct path in GraphRNN files.
