import argparse
import os
import sys

sys.path.append("..")
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from skimage import io
import random
import cv2
import time
######################################################
from DeviceSetting import device
from tfasr_model import Generator as srnet
from FeatureLoss_test import river_conterion, unet
from DEM_features import Slope_net, Aspect_net

seed = 10
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the high resolution image size')
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataroot', type=str,
                    default='/home/yifanzhang/PycharmProjects/PyTorch-SRGAN-master_1/process_data/cgan64_2/DEM_Train/',
                    help='path to dataset')
parser.add_argument('--netweight', type=str,
                    default='./tfasr_checkpoint/tfasr_099.pth',
                    help="path to generator weights (to continue training)")
out = "tfasr_test" + str(1)
parser.add_argument('--logfile', default='./EachDem-' + out + '.txt',
                    help="pre-training epoch times")

opt = parser.parse_args()

print(opt)


def write_file(filepath, target_tensor):
    with open(filepath, 'a') as af:
        ########################################
        af.write('ncols         64\n')
        af.write('nrows         64\n')
        af.write('xllcorner     -0.5\n')
        af.write('yllcorner     -0.5\n')
        af.write('cellsize      30\n')
        af.write('NODATA_value  -9999\n')
        ########################################
        num_rows, num_cols = target_tensor.shape
        for i in range(num_rows):
            for j in range(num_cols):
                af.write(str(target_tensor[i][j].item()) + ' ')
            af.write('\n')


dataset = datasets.ImageFolder(root=opt.dataroot)
assert dataset

model = srnet(16, opt.upSampling).to(device)

if opt.netweight != '':
    model.load_state_dict(torch.load(opt.netweight))
print(model)

content_criterion = nn.MSELoss().to(device)

high_res_real = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # high resolution dem
low_res = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # low resolution dem
base_min_list = torch.FloatTensor(opt.batchSize, 1).to(device)
base_max_list = torch.FloatTensor(opt.batchSize, 1).to(device)
high_river = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # river
fake_river = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
original_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # dem
fake_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
low_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
high_river_heatmap = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
fake_river_heatmap = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)

imgs_count = len(dataset.imgs)  # training dems count
batchSize_count = imgs_count // opt.batchSize  # number of batchSize in each epoch

model.eval()
random_ids = list(range(0, imgs_count))
random.shuffle(random_ids)  # shuffle these dems
print('river testing')
with torch.no_grad():
    for epoch in range(0, opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_river_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_dem_loss = 0.0
        mean_miou_0 = 0.0
        mean_miou_1 = 0.0
        mean_Miou = 0.0
        for i in range(0, batchSize_count):  # batchSize_count):
            # get a batchsize of rivers and dems
            for j in range(opt.batchSize):
                img_temp, _ = dataset.imgs[random_ids[i * opt.batchSize + j]]  # get one high-resolution image
                img_temp = io.imread(img_temp)
                H, W = img_temp.shape
                # low-resolution image
                low_img_temp = cv2.resize(img_temp, (H // opt.upSampling, W // opt.upSampling),
                                          interpolation=cv2.INTER_NEAREST)
                base_min = np.min(low_img_temp)
                base_max = np.max(low_img_temp)
                base_min_list[j] = torch.tensor(base_min)
                base_max_list[j] = torch.tensor(base_max)
                bicubic_high_img_temp = cv2.resize(low_img_temp, (H, W), interpolation=cv2.INTER_NEAREST)
                img_temp = torch.tensor(img_temp)  # 1*imagesize*imagesize
                original_dem[j] = img_temp
                low_dem[j] = torch.tensor(bicubic_high_img_temp)
                # 10 is a default value to keep safe
                img_temp = 2 * (img_temp - base_min) / (base_max - base_min + 10) - 1
                bicubic_high_img_temp = 2 * (bicubic_high_img_temp - base_min) / (base_max - base_min + 10) - 1
                high_res_real[j] = img_temp  # -1~1
                low_res[j] = torch.tensor(bicubic_high_img_temp)

            high_river = 1.0 * (F.sigmoid(unet(high_res_real.to(device))).detach().cpu() > 0.5).numpy().astype(
                np.float32)
            high_river_heatmap = torch.tensor(high_river).to(device)

            # Generate real and fake inputs
            high_res_real = Variable(high_res_real.to(device))
            high_res_fake = model(low_res.to(device)).to(device)

            generator_content_loss = content_criterion(high_res_fake.to(device), high_res_real.to(device))

            generator_river_loss, miou_0, miou_1, Miou = river_conterion(high_res_fake.to(device),
                                                                         high_river_heatmap.to(device),
                                                                         base_max_list.to(device))

            dem_loss = 0
            dem_loss_list = []
            for j in range(opt.batchSize):
                fake_dem[j] = (0.5 * (high_res_fake[j] + 1) * (base_max_list[j] - base_min_list[j] + 10) +
                               base_min_list[j]).to(device)
                dem_loss_temp = math.sqrt(content_criterion(original_dem[j], fake_dem[j]))
                dem_loss += dem_loss_temp
                dem_loss_list.append(dem_loss_temp)
            dem_loss = dem_loss / opt.batchSize

            slope_loss = 0
            slope_loss_list = []
            aspect_loss = 0
            aspect_loss_list = []
            high_slope = Slope_net(original_dem)
            fake_slope = Slope_net(fake_dem)
            high_aspect = Aspect_net(original_dem)
            fake_aspect = Aspect_net(fake_dem)

            for j in range(opt.batchSize):
                slope_loss_temp = math.sqrt(content_criterion(high_slope[j], fake_slope[j]))
                slope_loss_list.append(slope_loss_temp)
                slope_loss += slope_loss_temp

                aspect_loss_temp = math.sqrt(content_criterion(high_aspect[j], fake_aspect[j]))
                aspect_loss_list.append(aspect_loss_temp)
                aspect_loss += aspect_loss_temp

            slope_loss = slope_loss / opt.batchSize
            aspect_loss = aspect_loss / opt.batchSize
            ########## display each dem ##########
            errlog = open(opt.logfile, 'a')
            for j in range(opt.batchSize):
                errlog.write(
                    str(dem_loss_list[j]) + ',' + str(slope_loss_list[j]) + ',' + str(aspect_loss_list[j]) + ',' + str(
                        Miou[j]))
                errlog.write('\n')
            errlog.close()
            print(str(i) + ',' + str(batchSize_count))
