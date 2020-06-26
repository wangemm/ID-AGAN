import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Gen, Dis
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', type=str, default='./data/train/',
                    help='path to training dataset')
parser.add_argument('--test_root', type=str, default='./data/Set12/',
                    help='path to training dataset')
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30,
                    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--train_noise", nargs=2, type=int, default=[0, 55], help="Noise training interval")
parser.add_argument("--test_noise", type=float, default=25, help='noise level used on test set')
parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
parser.add_argument('--modelG', default='', help="path to netG (to continue training)")
parser.add_argument('--modelD', default='', help="path to modelD (to continue training)")
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build modelG
    resume_epoch = 0
    modelG = Gen(channels=opt.channels)
    modelG.apply(weights_init_kaiming)
    Gparam = sum(param.numel() for param in modelG.parameters())
    print('# modelG parameters:', Gparam)

    if opt.modelG != '':
        modelG.load_state_dict(torch.load(opt.modelG, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.modelG)['epoch']

    # Build modelD
    modelD = Dis(channels=opt.channels)
    modelG.apply(weights_init_kaiming)
    Dparam = sum(param.numel() for param in modelD.parameters())
    print('# modelD parameters:', Dparam)

    if opt.modelD != '':
        modelD.load_state_dict(torch.load(opt.modelD, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.modelD)['epoch']

    criterionBCE = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    modelG.cuda()
    modelD.cuda()
    criterionBCE.cuda()
    criterionMSE.cuda()

    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0
    label = Variable(label)

    # Optimizer
    optimizerG = optim.Adam(modelG.parameters(), lr=opt.lr)
    optimizerD = optim.Adam(modelD.parameters(), lr=opt.lr)

    # training
    step = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizerG.param_groups:
            param_group["lr"] = current_lr

        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # data
            img_train = data
            batch_size = img_train.size(0)

            noise = torch.zeros(img_train.size())
            stdN = np.random.uniform(opt.train_noise[0], opt.train_noise[1], size=noise.size()[0])
            for n in range(noise.size()[0]):
                sizeN = noise[0, :, :, :].size()
                noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            # train D
            fake = modelG(imgn_train)
            modelD.zero_grad()
            label.data.resize_(batch_size).fill_(real_label)
            for index1 in range(0, 255, 128):
                for index2 in range(0, 255, 128):
                    img_trainT = imgn_train[:, :, index1:index1 + 128, index1:index1 + 128]

                    output = modelD(img_trainT)
                    errD_real = criterionBCE(output, label)
                    errD_real.backward()

                    fakeT = fake[:, :, index1:index1 + 128, index1:index1 + 128]
                    label.data.fill_(fake_label)
                    output = modelD(fake.detach())
                    errD_fake = criterionBCE(output, label)
                    errD_fake.backward()

                    errD = errD_real + errD_fake
                    optimizerD.step()

            # train G
            modelG.train()
            modelG.zero_grad()
            optimizerG.zero_grad()
            label.data.fill_(real_label)
            errG_D = 0
            for index1 in range(0, 255, 128):
                for index2 in range(0, 255, 128):
                    fakeT = fake[:, :, index1:index1 + 128, index1:index1 + 128]
                    output = modelD(fakeT)
                    errG_D += criterionBCE(output, label) / 4.

            out_train = modelG(imgn_train)
            loss = criterionMSE(out_train, noise) + 0.01 * errG_D
            loss.backward()
            optimizerG.step()

            # results
            modelG.eval()
            denoise_image = torch.clamp(imgn_train - modelG(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(denoise_image, img_train, 1.)

            print("[epoch %d][%d/%d] Loss_G: %.4f PSNR_train: %.4f" % (
                epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            step += 1

        # log the images
        torch.save({'epoch': epoch + 1, 'state_dict': modelG.state_dict()}, 'model/modelG.pth')


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data(opt.train_root, opt.test_root)

    main()
