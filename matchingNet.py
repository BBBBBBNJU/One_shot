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
import torch.nn.functional as F
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

trainingImages = Variable(torch.FloatTensor(opt.shotnumber * opt.waynumber, 1, opt.imageSize, opt.imageSize))
trainingLables = Variable(torch.FloatTensor(opt.shotnumber * opt.waynumber, opt.waynumber))
testImages = Variable(torch.FloatTensor(opt.waynumber, 1, opt.imageSize, opt.imageSize))
testLables = Variable(torch.FloatTensor(opt.waynumber, opt.waynumber))

def generateOneTrainingSet() :
    targetSets = np.random.choice(trainingPaths, opt.waynumber, replace=False)
    for waynumber, eachfolder in enumerate(targetSets):
        allfiles = os.listdir(eachfolder)
        targetfiles = np.random.choice(allfiles, opt.shotnumber+1, replace=False)
        for shotnumber, eachfile in enumerate(targetfiles):
            imageFile = Image.open(eachfolder + '/' + eachfile).convert('RGB')
            temp_random = np.random.uniform()
            rotate = np.random.choice([0,90,270],1,replace=False)
            imageFile = imageFile.rotate(rotate[0])
            imageProcess = preprocess(imageFile)
            if eachfile != targetfiles[-1]:
                trainingImages[waynumber*opt.shotnumber + shotnumber, 0, :, :].data.copy_(imageProcess[0,:,:])
                trainingLables[waynumber*opt.shotnumber + shotnumber, waynumber].data.fill_(1.0)
            else:
                testImages[waynumber, 0, :, :].data.copy_(imageProcess[0,:,:])
                testLables[waynumber, waynumber].data.fill_(1.0)
    return trainingImages, trainingLables, testImages, testLables

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
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0
                    ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2),
                
                nn.Conv2d(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0
                    ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2),
                
                
                nn.Conv2d(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 0
                    ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 2),
                
                # nn.Conv2d(
                    # in_channels = 64,
                    # out_channels = 64,
                    # kernel_size = 3,
                    # stride = 1,
                    # padding = 0
                    # ),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace = True),
                # nn.MaxPool2d(kernel_size = 2),

                )
    def forward(self, x):
        output = self.model(x)
        return output.view(-1,64)

featureGenNet = featureNet()
featureGenNet.apply(weights_init)
featureGenNet.cuda()
trainIm, trainLabel,  testIm, testLabel = generateOneTrainingSet()
trainIm = trainIm.cuda()
trainLabel = trainLabel.cuda()
testIm = testIm.cuda()
testLabel = testLabel.cuda()

feature_train = featureGenNet(trainIm)
feature_test = featureGenNet(testIm)

class AttenLSTMCell(nn.Module):
    """ most code from https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py """
    def __init__(self, featureDim, use_bias=True):
        super(AttenLSTMCell, self).__init__()
        self.featureDim = featureDim
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(featureDim, 4 * featureDim))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(2*featureDim, 4 * featureDim))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * featureDim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_ih.data.set_(
            init.orthogonal(torch.FloatTensor(*self.weight_ih.size())))
        weight_hh_data = torch.eye(self.featureDim)
        weight_hh_data = weight_hh_data.repeat(2, 4)
        self.weight_hh.data.set_(weight_hh_data)
        self.bias.data.fill_(0)

    def forward(self, input_, hx):
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                                 split_size=self.featureDim, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {featureDim})'
        return s.format(name=self.__class__.__name__, **self.__dict__)



def attLSTM(nn.Module):
    def __init__(self, featureDim):
        super(attLSTM, self).__init__()
        self.featureDim = featureDim
        self.hidden_h = self.init_hidden(self.featureDim)
        self.hidden_c = self.init_hidden(self.featureDim)
        self.hidden_r = self.init_hidden(self.featureDim)
        self.attLSTMcell = AttenLSTMCell(self.featureDim)
    
    def init_hidden(self, hidden_dim):
        return Variable(torch.randn(1, self.hidden_dim))

    def updateHidden_r(self, trainF):
        att_ = F.sofrmax(torch.mm(self.hidden_h, torch.transpose(trainF, 0, 1)))
        att = torch.transpose(att_,0,1).repeat(1, self.featureDim)
        self.hidden_r = torch.sum(att * trainF, 0)
        
    def forward(x):
        trainF, testF = x
        trainNumber, _, _ = trainF.size()
        self.hidden_h , self.hidden_c= attLSTMcell(testF, 
                (torch.cat((self.hidden_h, self.hidden_r),1), self.hidden_c))
        self.hidden_h = self.hidden_h + testF
        self.updateHidden_r(trainF.view(trainNumber, -1))
        return self.hidden_h

def matchingNet(nn.Module):
    def __init__(self, trainingSampleNumber, testSampleNumber, featureDim):
        super(matchingNet, self).__init__()
        self.featureDim = featureDim
        self.featureGenNet = featureNet()
        self.conEmbed_G = nn.LSTM(self.featureDim, self.featureDim, bidirectional = True)
        self.conEmbed_F = attLSTM(self.featureDim)
        self.hidden_G = self.init_hidden_G(self.featureDim)
        self.cond_feature_train = Variable(torch.FloatTensor(trainingSampleNumber, 1, self.featureDim))
        self.cond_feature_test = Variable(torch.FloatTensor(testSampleNumber, 1, self.featureDim))
    def init_hidden_G(self, hidden_dim):
         return (Variable(torch.randn(2, 1, self.hidden_dim)),
                 Variable(torch.randn(2, 1, self.hidden_dim)))
   
    def getCondEmbed_G(self, feature_train):
        totalOutput, self.hidden_G = self.conEmbed_G(feature_train, self.hidden_G)
        sampleNumber, _, doubleHidden_dim = totalOutput.size()
        for i in range(sampleNumber):
            self.cond_feature_train[i,:,:].data.fill_(totalOutput[i,:,0:doubleHidden_dim/2] + 
                    totalOutput[trainingSampleNumber - i,:,doubleHidden_dim/2:None] +
                    feature_train[i,:,:])
    
    def getCondEmbed_F(self, feature_train, feature_test):
        sampleNumber, _, _= feature_test
        for i in range(sampleNumber):
            self.cond_feature_test[i,:,:].data.fill_()

    def forward(x):
        trainIm, trainLabel, testIm, testLabel = x
        feature_train = featureGenNet(trainIm)
        feature_test = featureGenNet(testIm)
        getCondEmbed_G(feature_train)
        
