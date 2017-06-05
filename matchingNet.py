from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='omniglot', help ='path to dataset')
parser.add_argument('--shotnumber', type=int, default=1, help='shot number')
parser.add_argument('--waynumber', type=int, default=5, help='way number')
parser.add_argument('--imageSize', type=int, default=28, help='input image size')
opt = parser.parse_args()
print(opt)

allCharacters = [x[0] for x in os.walk(opt.dataroot) if x[0][-1].isdigit()]
random.shuffle(allCharacters)
trainingPaths = allCharacters[:1200]
testPaths = allCharacters[1200:None]
preprocess = transforms.Compose([
    transforms.Scale(opt.imageSize),
    transforms.ToTensor(),
    ])

trainingImages = Variable(torch.FloatTensor(opt.shotnumber * opt.waynumber, 3, opt.imageSize, opt.imageSize))
trainingLables = Variable(torch.FloatTensor(opt.shotnumber * opt.waynumber, opt.waynumber))
testImages = Variable(torch.FloatTensor(opt.waynumber, 3, opt.imageSize, opt.imageSize))
testLables = Variable(torch.FloatTensor(opt.waynumber, opt.waynumber))

def generateOneTrainingSet() :
    targetSets = np.random.choice(trainingPaths, opt.waynumber, replace=False)
    lableCount = 0
    for waynumber, eachfolder in enumerate(targetSets):
        allfiles = os.listdir(eachfolder)
        targetfiles = np.random.choice(allfiles, opt.shotnumber+1, replace=False)
        for shotnumber, eachfile in enumerate(targetfiles):
            imageFile = Image.open(eachfolder + '/' + eachfile).convert('RGB')
            temp_random = np.random.uniform()
            rotate = np.random.choice([0,90,270],1,replace=False)
            imageFile = imageFile.rotate(rotate[0])
            if eachfile != targetfiles[-1]:
                trainingImages[waynumber*opt.shotnumber + shotnumber, :, :, :].data.copy_(preprocess(imageFile))
                trainingLables[waynumber*opt.shotnumber + shotnumber, waynumber].data.fill_(1.0)
            else:
                testImages[waynumber, :, :, :].data.copy_(preprocess(imageFile))
                testLables[waynumber, waynumber].data.fill_(1.0)
        lableCount += 1
    return trainingImages, trainingLables, testImages, testLables
    # return Variable(trainingImages).cuda(), Variable(trainingLables).cuda(), Variable(testImages).cuda(), Variable(testLables).cuda()

def weights_init(m):
	if isinstance(m, nn.ConvTranspose2d):
		m.weight.data.normal_(mean=0,std=0.002)
	if isinstance(m, nn.BatchNorm2d):
		m.weight.data.normal_(1.0, 0.002)
		m.bias.data.fill_(0.0)

class featureNet(nn.Module):
    def __init__(self):
        super(featureNet, self).__init__()
        self.model = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels = 3,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0
                    ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(2),
                
                nn.ConvTranspose2d(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0
                    ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(2),
                
                
                nn.ConvTranspose2d(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0
                    ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(2),
                
                nn.ConvTranspose2d(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0
                    ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(2)
                )
    def forward(self, x):
        output = self.model(x)
        return output

featureGenNet = featureNet()
featureGenNet.apply(weights_init)
featureGenNet.cuda()
trainIm, _,_,_ = generateOneTrainingSet()
trainIm = trainIm.cuda()
output = featureGenNet(trainIm)
print (output)





