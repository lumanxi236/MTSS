"""
General test script for HiSD. 
"""

# if type is 'latent-guided', you need to specifc 'attribute' and 'seed' (None means random).
# Otherwise if type is 'reference-guided', you need to specifc the 'reference' file path.
steps = [
#     {'type': 'latent-guided', 'tag': 0, 'attribute': 0, 'seed': None},
# {'type': 'latent-guided', 'tag': 0, 'attribute': 1, 'seed': None},
# {'type': 'latent-guided', 'tag': 0, 'attribute': 2, 'seed': None},
# {'type': 'latent-guided', 'tag': 0, 'attribute': 3, 'seed': None},
# {'type': 'latent-guided', 'tag': 0, 'attribute': 4, 'seed': None},
# {'type': 'latent-guided', 'tag': 0, 'attribute': 5, 'seed': None},
# {'type': 'latent-guided', 'tag': 0, 'attribute': 6, 'seed': None},
# {'type': 'latent-guided', 'tag': 0, 'attribute': 7, 'seed': None},
{'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_72_Moroccan_male_angry_frontal.jpg'},
    {'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_45_Moroccan_male_contemptuous_frontal.jpg'},
    {'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_72_Moroccan_male_disgusted_frontal.jpg'},
    {'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_20_Caucasian_male_fearful_right.jpg'},
    {'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_71_Caucasian_male_happy_frontal.jpg'},
    {'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_70_Moroccan_male_neutral_right.jpg'},
    {'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_73_Moroccan_male_sad_frontal.jpg'},
    {'type': 'reference-guided', 'tag': 0,
     'reference': './Rafd090_68_Moroccan_male_surprised_left.jpg'},
    # {'type': 'latent-guided', 'tag': 1, 'attribute': 0, 'seed': None},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_72_Moroccan_male_angry_frontal.jpg'},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_45_Moroccan_male_contemptuous_frontal.jpg'},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_72_Moroccan_male_disgusted_frontal.jpg'},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_20_Caucasian_male_fearful_right.jpg'},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_71_Caucasian_male_happy_frontal.jpg'},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_70_Moroccan_male_neutral_right.jpg'},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_73_Moroccan_male_sad_frontal.jpg'},
    # {'type': 'reference-guided', 'tag': 0,
    #  'reference': './Rafd090_68_Moroccan_male_surprised_left.jpg'},
]

from utils import get_config
from trainer import HiSD_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as Fu
import csv


parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)

opts = parser.parse_args()



os.makedirs(opts.output_path, exist_ok=True)

config = get_config(opts.config)
noise_dim = config['noise_dim']
trainer = HiSD_Trainer(config)
state_dict = torch.load(opts.checkpoint)
trainer.models.gen.load_state_dict(state_dict['gen_test'])
trainer.models.gen.cuda(0)

E = trainer.models.gen.encode
T = trainer.models.gen.translate
G = trainer.models.gen.decode
M = trainer.models.gen.map
F = trainer.models.gen.extract
C = trainer.models.gen.classifer
A = trainer.models.gen.afiu
trainer.models.gen.extractors = trainer.models.gen.extractors.eval()
# trainer.models.gen.mappers = trainer.models.gen.mappers.eval()
filename = time.time()
transform = transforms.Compose([
# transforms.CenterCrop((1080, 1080)),
                        transforms.Resize(config['new_size']),
                        # transforms.CenterCrop((100, 100)),
                        #         transforms.Resize(config['new_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if os.path.isfile(opts.input_path):
    inputs = [opts.input_path]
else:
    inputs = [os.path.join(opts.input_path, file_name) for file_name in os.listdir(opts.input_path)]

with torch.no_grad():
    nums = 0
    for input in inputs:
        # input = os.path.join("/NAS_REMOTE/pzb/GANimation/dataset/imgs_crop/", input[0])
        xs = []
        nums += 1
        x = transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda(0)
        xs.append(x)
        # x = torch.cat(xs, dim=0)
        c, f_s = E(x)
        _, f_r = F(x, 0)
        _, f_s = A(f_r, f_s)


        c_trg = c
        x_sample = [x]
        result_path = os.path.join(opts.output_path, 'real')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        vutils.save_image(((x + 1) / 2).data,
                          os.path.join(result_path, f'{os.path.basename(input)}_output.jpg'), padding=0)

        for j in range(len(steps)):
            step = steps[j]
            if step['type'] == 'latent-guided':
                if step['seed'] is not None:
                    torch.manual_seed(step['seed'])
                    torch.cuda.manual_seed(step['seed'])

                z = torch.randn(1, noise_dim).cuda(0)
                s_trg = M(z, step['tag'], step['attribute'])

            elif step['type'] == 'reference-guided':
                reference = transform(Image.open(step['reference']).convert('RGB')).unsqueeze(0).cuda(0)
                s_trg, _ = F(reference, step['tag'])

            c_trg_ = T(c_trg, s_trg, step['tag'])

            x_trg = G(c_trg_, f_s)
            result_path = os.path.join(opts.output_path, 'a{}'.format(j + 1))
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            vutils.save_image(((x_trg + 1) / 2).data,
                              os.path.join(result_path, f'{os.path.basename(input)}_output.jpg'), padding=0)


