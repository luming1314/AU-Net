# #!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=0,1  torchrun --nproc_per_node=2  --master_port=29512 ./scripts/train.py \
                                                                            --dataroot=./datasets/train/NirScene \
                                                                            --n_ep=1200 \
                                                                            --n_ep_decay=200 \
                                                                            --resume=./checkpoint/AU-Net.pth \
                                                                            --stage=RF \
                                                                            --batch_size=8 \
                                                                            --name=AU-Net \
                                                                            --first_conv_class=ODConv \
                                                                            --tensorboardX_path=./logs/AU_Net \
                                                                            --use_style
"""

Argument Descriptions:

--dataroot                                   # Path to the training dataset
--n_ep                                       # Total number of training epochs
--n_ep_decay                                 # Epoch to start decaying the learning rate
--resume                                     # Initial weights for AU-Net
--stage                                      # Flags for registration and fusion
--batch_size                                 # Training batch size
--name                                       # Name of the model
--first_conv_class                           # Enable dynamic convolution
--tensorboardX_path                          # Path for saving logs
--use_style                                  # Use pseudo-infrared or pseudo-visible images

"""