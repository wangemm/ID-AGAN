import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Gen
from utils import *
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--dataroot', type=str, default='./data/Set12', help='path to dataset')
parser.add_argument('--modelG', type=str, default='./model/modelG.pth',
                    help="path to modelG weights (to continue training)")
parser.add_argument("--noise_ratio", type=float, default=15, help='noise level used on test set')
parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
opt = parser.parse_args()


def normalize(data):
    return data / 255.


def variable_to_cv2_image(varim):
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :] * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res * 255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res


def main():
    # Build model

    print('Loading model ...\n')
    model = Gen(channels=1)
    model.cuda()
    model.load_state_dict(torch.load(opt.modelG, map_location=lambda storage, location: storage)['state_dict'])
    model.eval()
    # load data info
    print('Loading data info ...\n')
    types = ('*.bmp', '*.png', '*.jpg')
    files = []
    for im in types:
        files.extend(glob.glob(os.path.join(opt.dataroot, im)))
    files.sort()
    # process data
    psnr_test = 0
    ssim_test = 0
    results_psnr = []
    results_ssim = []
    it = 0
    for f in files:
        # image
        if opt.channels == 3:
            Img = cv2.imread(f)
            Img = (cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            Img = np.expand_dims(Img, 0)

        Img = np.expand_dims(Img, 0)

        if Img.shape[2] % 32 != 0:
            Img = Img[:, :, :Img.shape[2] - Img.shape[2] % 32, :]

        if Img.shape[3] % 32 != 0:
            Img = Img[:, :, :, :Img.shape[3] - Img.shape[3] % 32]

        Img = normalize(Img)
        ISource = torch.Tensor(Img)
        N, C, H, W = ISource.size()

        dtype = torch.cuda.FloatTensor
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.noise_ratio / 255.)

        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.type(dtype)), Variable(INoisy.type(dtype))
        starttime = datetime.datetime.now()
        Out = torch.clamp(INoisy - model(INoisy), 0., 1.)
        endtime = datetime.datetime.now()

        psnr = batch_PSNR(Out, ISource, 1.)
        ssim = batch_SSIM(Out, ISource, 1.)
        psnrniose = batch_PSNR(INoisy, ISource, 1.)
        ssimniose = batch_SSIM(INoisy, ISource, 1.)
        results_psnr.append(psnr)
        results_ssim.append(ssim)
        psnr_test += psnr
        ssim_test += ssim
        print("%s PSNR %f SSIM %f" % (f, psnr, ssim))
        real = variable_to_cv2_image(ISource)
        denoise = variable_to_cv2_image(Out)
        noise = variable_to_cv2_image(INoisy)
        cv2.imwrite("./test-results/noise/%d.png" % it, noise)
        cv2.imwrite("./test-results/denoise/%d.png" % it, denoise)
        cv2.imwrite("./test-results/real/%d.png" % it, real)
        it += 1
    psnr_test /= len(files)
    ssim_test /= len(files)
    print("\nPSNR on test data %f SSIM on test data %f" % (psnr_test, ssim_test))
    std_psnr = np.std(results_psnr)
    std_ssim = np.std(results_ssim)
    print("\nPSNRstd on test data %f SSIMstd on test data %f" % (std_psnr, std_ssim))
    print(endtime - starttime)


if __name__ == "__main__":
    main()
