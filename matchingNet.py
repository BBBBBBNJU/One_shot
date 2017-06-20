from __future__ import print_function
from torch.nn import init
import time
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
parser.add_argument('--lr', type=float, default=0.0002,help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for momentum')
parser.add_argument('--trainround', type=int, default=200, help='training rounds number')
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

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Direct copy from pytorch orginal source code
    No idea why it can not be imported"""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

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
                )
        
    def forward(self, x):
        output = self.model(x)
        return output.view(-1,64).unsqueeze(1)


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


class attLSTM(nn.Module):
    def __init__(self, featureDim):
        super(attLSTM, self).__init__()
        self.featureDim = featureDim
        self.hidden_h = self.init_hidden(self.featureDim)
        self.hidden_c = self.init_hidden(self.featureDim)
        self.hidden_r = self.init_hidden(self.featureDim)
        self.attLSTMcell = AttenLSTMCell(self.featureDim)
    
    def init_hidden(self, hidden_dim):
        return Variable(torch.randn(1, hidden_dim)).cuda()

    def updateHidden_r(self, trainF):
        att_ = F.softmax(torch.mm(self.hidden_h, torch.transpose(trainF, 0, 1)))
        att = torch.transpose(att_,0,1).repeat(1, self.featureDim)
        self.hidden_r = torch.sum(att * trainF, 0)
        
    def forward(self, x):
        (trainF, testF) = x
        trainNumber, _, _ = trainF.size()
        
        self.hidden_h , self.hidden_c= self.attLSTMcell(testF, 
                (torch.cat((self.hidden_h, self.hidden_r),1), self.hidden_c))
        self.hidden_h = self.hidden_h + testF
        self.updateHidden_r(trainF.view(trainNumber, -1))
        return self.hidden_h

class matchingNet(nn.Module):
    def __init__(self, trainingSampleNumber, testSampleNumber, featureDim):
        super(matchingNet, self).__init__()
        self.trainingSampleNumber = trainingSampleNumber
        self.testSampleNumber = testSampleNumber
        self.featureDim = featureDim
        self.featureGenNet = featureNet()
        self.conEmbed_G = nn.LSTM(self.featureDim, self.featureDim, bidirectional = True)
        self.conEmbed_F = attLSTM(self.featureDim)
        self.hidden_G = self.init_hidden_G(self.featureDim)

    def init_hidden_G(self, hidden_dim):
         return (Variable(torch.randn(2, 1, hidden_dim)).cuda(),
                 Variable(torch.randn(2, 1, hidden_dim)).cuda())
   
    def getCondEmbed_G(self, feature_train):
        totalOutput, self.hidden_G = self.conEmbed_G(feature_train, self.hidden_G)
        sampleNumber, _, doubleHidden_dim = totalOutput.size()
        cond_feature_train = Variable(torch.FloatTensor(self.trainingSampleNumber, 1, self.featureDim)).cuda()
        for i in range(sampleNumber):
            cond_feature_train[i,:,:] = totalOutput[i,:,0:doubleHidden_dim/2] + \
            totalOutput[self.trainingSampleNumber - 1 - i,:,doubleHidden_dim/2:None] + \
            feature_train[i,:,:]
        return cond_feature_train

    def getCondEmbed_F(self, feature_train, feature_test):
        sampleNumber, _, _= feature_test.size()
        cond_feature_test = Variable(torch.FloatTensor(self.testSampleNumber, 1, self.featureDim)).cuda()
        for i in range(sampleNumber):
            cond_feature_test[i,:,:] = self.conEmbed_F((feature_train, feature_test[i,:,:]))
        return cond_feature_test

    def predictLabel(self, vec1, mat1, label1):
        sampleNumber,_ ,_ = mat1.size()
        mat2 = vec1.repeat(sampleNumber, 1)
        mat1 = torch.squeeze(mat1, 1)
        temp_sim = cosine_similarity(mat2, mat1, 1) 
        temp_att = F.softmax(temp_sim)
        temp_result = torch.mm(torch.unsqueeze(temp_att,0), label1)  
        return temp_result
    
    def getLabel(self, cond_feature_train, cond_feature_test, trainLabel):
        sampleNumber = cond_feature_test.size()[0]
        temp_predictLabel = Variable(torch.FloatTensor(sampleNumber, opt.waynumber)).cuda()
        for i in range(sampleNumber):
            temp_predictLabel[i,:] = self.predictLabel(cond_feature_test[i,:,:], cond_feature_train, trainLabel) 
        return temp_predictLabel

    def forward(self, x):
        trainIm, trainLabel, testIm, testLabel = x
        feature_train = self.featureGenNet(trainIm)
        feature_test = self.featureGenNet(testIm)
        cond_feature_train = self.getCondEmbed_G(feature_train)
        cond_feature_test = self.getCondEmbed_F(cond_feature_train, feature_test)
        return self.getLabel(cond_feature_train, cond_feature_test, trainLabel)

ourMatchingNet = matchingNet(opt.waynumber * opt.shotnumber, opt.waynumber, 64)
ourMatchingNet.cuda()
print (ourMatchingNet)

def generateOneTrainingSet() :
    trainingImages = Variable(torch.FloatTensor(opt.shotnumber * opt.waynumber, 1, opt.imageSize, opt.imageSize)).cuda()
    trainingLables = Variable(torch.FloatTensor(opt.shotnumber * opt.waynumber, opt.waynumber)).cuda()
    testImages = Variable(torch.FloatTensor(opt.waynumber, 1, opt.imageSize, opt.imageSize)).cuda()
    testLables = Variable(torch.LongTensor(opt.waynumber)).cuda()
    targetSets = np.random.choice(trainingPaths, opt.waynumber, replace=False)
    trainingLables.data.zero_()
    testLables.data.zero_()
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
                trainingLables[waynumber*opt.shotnumber + shotnumber, waynumber] = 1.0
            else:
                testImages[waynumber, 0, :, :].data.copy_(imageProcess[0,:,:])
                testLables[waynumber] = waynumber
    return trainingImages, trainingLables,  testImages, testLables


m = nn.LogSoftmax()
loss_function = nn.NLLLoss()
optimizer_matchNet = optim.Adam(ourMatchingNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
for iterNumber in range(opt.trainround):
    start = time.time()
    ourMatchingNet.zero_grad()
    trainingImages, trainingLables,  testImages, testLables = generateOneTrainingSet()
    output = ourMatchingNet((trainingImages, trainingLables,  testImages, testLables))
    error = loss_function(m(output), testLables)
    error.backward(retain_variables = True)
    optimizer_matchNet.step()
    stop = time.time()
    print ('[%d/%d] round, loss: %.4f, time %.4f' % (iterNumber, opt.trainround, error.data[0], stop-start))

