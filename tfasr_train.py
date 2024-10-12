import argparse # For parsing command-line arguments
import os # for interacting with the operating system
import sys # manipulation of the Python runtime envi

sys.path.append("..")
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler # Provides learning rate schedulers
import torch.nn as nn # Contains modules and functions for neural network building blocks
from torch.autograd import Variable # Provides automatic differentiation for all operations on Tensors
import numpy as np # A library for numerical operations on arrays
import torchvision
from torch.nn import functional as F
import torchvision.datasets as datasets # Provides standard datasets (e.g., ImageNet, CIFAR-10)
import torchvision.transforms as transforms # Provides common image transformations
from skimage import io # For reading and writing images
import random
import cv2 # OpenCV library for computer vision tasks
import time
######################################################
from DeviceSetting import device
from tfasr_model import Generator as srnet
from FeatureLoss import river_conterion, unet
from DEM_features import Slope_net

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# parse command-line arguments for configuring the training process of the model
parser = argparse.ArgumentParser() # Initializes a new argument parser object
parser.add_argument('-- workers', type=int, default=1, help='number of data loading workers')
# --workers: Specifies the number of subprocesses to use for data loading. Default is 1.
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
# --batchSize: Determines the number of samples per batch to load. Default is 16.
parser.add_argument('--imageSize', type=int, default=64, help='the high resolution image size')
# --imageSize: Defines the size of the high-resolution images. Default is 64.
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
# --upSampling: Specifies the factor by which the low-resolution image will be scaled up to create the high-resolution image. Default is 4.
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
# --nEpochs: The number of complete passes through the training dataset. Default is 100.
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate for generator')
# --lr: Learning rate for the optimizer used to train the generator model. Default is 0.00001.
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
# --nGPU: The number of GPUs to be used for training. Default is 1.
parser.add_argument('--dataroot', type=str,
                    default='D:/ML_DL_NLP/New_project/New folder (3)/TFaSR_code/DEM_Train/',
                    help='path to dataset')
# --dataroot: The root directory path where the training dataset is located. Default is a specified path.
parser.add_argument('--netweight', type=str,
                    default='',
                    help="path to generator weights (to continue training)")
# --netweight: path to pre-trained weights for generator model, useful for resuming training from a checkpoint. Default = empty string (ie no pre-trained weights).
out = "tfasr_" + str(1)
parser.add_argument('--river_weight', type=float, default=1, help='')  # α = weight for river loss component in overall loss calculation. Default is 1.
parser.add_argument('--slope_weight', type=float, default=0, help='')  # β = weight for slope loss component in overall loss calculation. Default is 0
parser.add_argument('--out', type=str, default='./checkpoints/' + out,
                    help='folder to output model checkpoints')
# --out: The directory where model checkpoints will be saved. Default is a dynamically generated path ./checkpoints/tfasr_1
parser.add_argument('--logfile', default='./errlog-' + out + '.txt',
                    help="pre-training epoch times")
# --logfile: The path to the log file where errors and other information for each epoch will be recorded. Default is ./errlog-tfasr_1.txt.
parser.add_argument('--ave_logfile', default='./ave-' + out + '.txt',
                    help="pre-training epoch times")
# --ave_logfile: The path to the log file where average error values for each epoch will be recorded. Default is ./ave-tfasr_1.txt

opt = parser.parse_args()
# Parses the arguments provided via the command line and stores them in the opt variable.

print(opt)
# Prints the parsed arguments to the console for verification.

# output directory creation
try:
    os.makedirs(opt.out)
except OSError:
    pass


def write_file(filepath, target_tensor):
    # filepath: The path to the file where the tensor data will be written.
    # target_tensor: The tensor whose data will be written to the file. 
    with open(filepath, 'a') as af: # opens the file specified by filepath in append mode, af is the file object that will be used to write to the file
        num_rows, num_cols = target_tensor.shape # gets the dimensions of the target_tensor
        for i in range(num_rows):
            for j in range(num_cols):
                af.write(str(target_tensor[i][j].item()) + ',') # retrieves the scalar value from the tensor at position [i][j] and converts it to a Python scalar
                # converts the scalar value to a string and writes it to the file, followed by a comma
            af.write('\n')
            # writes a newline character to the file to move to the next line


dataset = datasets.ImageFolder(root=opt.dataroot)
assert dataset # to check if the dataset object was created successfully

model = srnet(16, opt.upSampling).to(device)

if opt.netweight != '':
# checks if the --netweight argument is provided and not an empty string. 
# If it's not empty, it means there is a pre-trained model weight file specified.
    model.load_state_dict(torch.load(opt.netweight))
# torch.load(opt.netweight) loads the pre-trained model weights from the file path specified by opt.netweight
# model.load_state_dict() updates model’s para with loaded weights. => can resume training from where you left off/use a pre-trained model.
print(model)
# prints the model architecture to the console

optimiser = optim.Adam(model.parameters(), lr=opt.lr)
# initializes the optimizer for the model
content_criterion = nn.MSELoss().to(device)
#  initializes the loss function used for training

high_res_real = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # high resolution dem
low_res = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # low resolution dem
base_min_list = torch.FloatTensor(opt.batchSize, 1).to(device)
base_max_list = torch.FloatTensor(opt.batchSize, 1).to(device)
high_river = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # river
fake_river = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
original_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # dem
fake_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
high_river_heatmap = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
fake_river_heatmap = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)

imgs_count = len(dataset.imgs)  # training dems count
print(imgs_count)
batchSize_count = imgs_count // opt.batchSize  # number of batchSize in each epoch

model.train()
random_ids = list(range(0, imgs_count))
random.shuffle(random_ids)  # shuffle these dems
print('river training')
for epoch in range(0, opt.nEpochs):
    if epoch >= 80:
        opt.river_weight = 1e-3
    mean_generator_content_loss = 0.0
    mean_generator_river_loss = 0.0
    mean_generator_slope_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_dem_loss = 0.0
    mean_miou_0 = 0.0
    mean_miou_1 = 0.0
    mean_Miou = 0.0
    for i in range(batchSize_count):
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
            # 10 is a default value to keep safe
            img_temp = 2 * (img_temp - base_min) / (base_max - base_min + 10) - 1
            bicubic_high_img_temp = 2 * (bicubic_high_img_temp - base_min) / (base_max - base_min + 10) - 1
            high_res_real[j] = img_temp  # -1~1
            low_res[j] = torch.tensor(bicubic_high_img_temp)

        high_river = 1.0 * (F.sigmoid(unet(high_res_real.to(device))).detach().cpu() > 0.5).numpy().astype(np.float32)
        high_river_heatmap = torch.tensor(high_river).to(device)

        # Generate real and fake inputs
        high_res_real = Variable(high_res_real.to(device))
        high_res_fake = model(low_res.to(device)).to(device)

        ######### Train generator #########
        optimiser.zero_grad()

        high_slope = Slope_net(high_res_real)
        fake_slope = Slope_net(high_res_fake)
        generator_slope_loss = content_criterion(high_slope, fake_slope)

        generator_content_loss = content_criterion(high_res_fake.to(device), high_res_real.to(device))

        generator_river_loss, miou_0, miou_1, Miou = river_conterion(high_res_fake.to(device),
                                                                     high_river_heatmap.to(device),
                                                                     base_max_list.to(device))

        generator_total_loss = generator_content_loss + opt.slope_weight * generator_slope_loss + opt.river_weight * generator_river_loss

        generator_total_loss.backward()
        optimiser.step()

        dem_loss = 0
        for j in range(opt.batchSize):
            fake_dem[j] = (0.5 * (high_res_fake[j] + 1) * (base_max_list[j] - base_min_list[j] + 10) +
                           base_min_list[j]).to(device)
            dem_loss += math.sqrt(content_criterion(original_dem[j], fake_dem[j]))
        dem_loss = dem_loss / opt.batchSize
        ######### Status and display #########
        sys.stdout.write(
            '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f' % (
                epoch, opt.nEpochs, i, batchSize_count,
                generator_content_loss.data,
                generator_river_loss.data,
                generator_slope_loss.data,
                generator_total_loss.data,
                dem_loss, miou_0, miou_1, Miou))
        errlog = open(opt.logfile, 'a')
        errlog.write(
            '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f' % (
                epoch, opt.nEpochs, i, batchSize_count,
                generator_content_loss.data,
                generator_river_loss.data,
                generator_slope_loss.data,
                generator_total_loss.data,
                dem_loss, miou_0, miou_1, Miou))
        errlog.close()
        ######## mean value of each epoch ##############
        mean_generator_content_loss += generator_content_loss.data
        mean_generator_river_loss += generator_river_loss.data
        mean_generator_slope_loss += generator_slope_loss.data
        mean_generator_total_loss += generator_total_loss.data
        mean_dem_loss += dem_loss
        mean_miou_0 += miou_0
        mean_miou_1 += miou_1
        mean_Miou += Miou

    mean_generator_content_loss = mean_generator_content_loss / batchSize_count
    mean_generator_river_loss = mean_generator_river_loss / batchSize_count
    mean_generator_slope_loss = mean_generator_slope_loss / batchSize_count
    mean_generator_total_loss = mean_generator_total_loss / batchSize_count
    mean_dem_loss = mean_dem_loss / batchSize_count
    mean_miou_0 = mean_miou_0 / batchSize_count
    mean_miou_1 = mean_miou_1 / batchSize_count
    mean_Miou = mean_Miou / batchSize_count

    sys.stdout.write(
        '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f\n' % (
            epoch, opt.nEpochs, i, batchSize_count,
            mean_generator_content_loss, mean_generator_river_loss, generator_slope_loss,
            mean_generator_total_loss,
            mean_dem_loss, mean_miou_0, mean_miou_1, mean_Miou))
    ave_errlog = open(opt.ave_logfile, 'a')
    ave_errlog.write(
        '\r[%d/%d][%d/%d] Generator_Loss (Content/river/slope/Total/demloss/miou_0/miou_1/Miou ): %.8f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f\n' % (
            epoch, opt.nEpochs, i, batchSize_count,
            mean_generator_content_loss, mean_generator_river_loss, generator_slope_loss,
            mean_generator_total_loss,
            mean_dem_loss, mean_miou_0, mean_miou_1, mean_Miou))
    ave_errlog.close()

    # Do checkpointing
    torch.save(model.state_dict(), '%s/generator_final_%03d.pth' % (opt.out, epoch))
