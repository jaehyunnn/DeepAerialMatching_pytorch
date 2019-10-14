from __future__ import print_function, division
import torch
from collections import OrderedDict
import sys
from tqdm import tqdm
from time import time

from util.torch_util import BatchTensorToVars
from geotnf.point_tnf import *

def train(epoch,model,loss_fn,optimizer,dataloader,pair_generation_tnf,use_cuda=True,log_interval=50):
    model.train()
    train_loss = 0
    start_time = time()
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta_AB, theta_BA, theta_AC, theta_CA = model(tnf_batch)
        loss = loss_fn(theta_AB,theta_BA,theta_AC,theta_CA,tnf_batch['theta_GT'])
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        #sys.stderr.write('Learning batch [%d]\n'%(batch_idx+1))
        #sys.stderr.flush()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss.data))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f} --- {:.2f}s'.format(train_loss, (time()-start_time)))
    return train_loss

def test(model,loss_fn,dataloader,pair_generation_tnf,use_cuda=True):
    model.eval()
    test_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta_AB, theta_BA, theta_AC, theta_CA = model(tnf_batch)
        loss = loss_fn(theta_AB,theta_BA, theta_AC, theta_CA,tnf_batch['theta_GT'])
        test_loss += loss.data.cpu().numpy()

    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss
