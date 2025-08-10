# #!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

python -u ./scripts/test.py \
          --dataset_path=./datasets/test \
          --mode='Reg&Fusion' \
          --dataset_name=Simple \
          --model_path=./checkpoint/AU-Net.pth \
          --save_path=Reg_Fusion_warp \
          --method_name=AU-Net \
          --first_conv_class=ODConv \
          --save_root=./results

"""

Argument Descriptions:

--dataset_path               # Specifies the directory containing the test dataset
--mode                       # Selects the task mode: Reg for registration only, Fusion for fusion only, and Reg&Fusion for joint registration and fusion.
--dataset_name               # Indicates the dataset name to be used for testing.
--model_path                 # Provides the file path to the pre-trained AU-Net model.
--save_path                  # Defines the subdirectory within the results folder where outputs will be stored.
--method_name                # Specifies the method name for logging and output organization.
--first_conv_class           # Enables dynamic convolution in the model (set flag to activate).
--save_root                  # Specifies the root directory where all output results will be saved.

"""
