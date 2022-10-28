# Multi task-based facial expression synthesis with supervision learning and feature disentanglement of image style
This is the code repository for our paper "Multi task-based facial expression synthesis with supervision learning and feature disentanglement of image style".

# Code
+ Pytorch
  
  Torch 1.6.0 or higher and Python 3.6 or higher are required.

+ Prepare
  
  You need to divide the RaFD into 8 expressions and prepare the corresponding txt file, put the txt file into ./datasets

  Download the model into ./checkpoint
  
  Put the image to be tested into ./input

+ Test MTSS
```
python core/test.py --config configs/RaFD.yaml
--checkpoint ./checkpoints/gen_00200000.pt
--input_path ./input
--output_path ./output
```

The pre-trained model on AffectNet dataset is available in [link](https://drive.google.com/drive/folders/1cz9pVkyrFrENOLxP6gGBqtiDj7MQxI2O?usp=sharing)

If you find our code or paper useful, please cite us.
