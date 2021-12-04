# Maximum Mean Discrepancy function MMD() courtesy of: https://www.kaggle.com/onurtunali/maximum-mean-discrepancy

import glob
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StatisticalDetection:

    def __init__(self, folder1path, folder2path):
        self.hist_1 = np.zeros([256, 1], dtype=float)
        self.hist_2 = np.zeros([256, 1], dtype=float)
        # Read all the images in as grayscale
        filenames1 = glob.glob(folder1path)
        filenames1.sort()
        filenames2 = glob.glob(folder2path)
        filenames2.sort()
        self.images_1 = [cv2.imread(imgfile, 0) for imgfile in filenames1]
        self.images_2 = [cv2.imread(imgfile, 0) for imgfile in filenames2]
        # Init ORB object
        self.orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

    def findDistributions(self):

        # Perform ORB feature extraction (images_1 and images_2 are same length)
        for i in range(len(self.images_1)):
            # Image from set 1 first
            keypoints = self.orb.detect(self.images_1[i], None)
            keypoints, _ = self.orb.compute(self.images_1[i], keypoints)
            kp_img = cv2.drawKeypoints(self.images_1[i], keypoints, None, color=(0, 255, 0), flags=0)
            hist = cv2.calcHist([kp_img], [0], None, [256], [0, 256])
            np.add(self.hist_1, hist, out=self.hist_1)
            # Repeat for image from second set
            keypoints = self.orb.detect(self.images_2[i], None)
            keypoints, _ = self.orb.compute(self.images_2[i], keypoints)
            kp_img = cv2.drawKeypoints(self.images_2[i], keypoints, None, color=(0, 255, 0), flags=0)
            hist = cv2.calcHist([kp_img], [0], None, [256], [0, 256])
            np.add(self.hist_2, hist, out=self.hist_2)
            
        '''
        plt.plot(self.hist_1)
        plt.show()
        plt.plot(self.hist_2)
        plt.show()
        '''

    def MMD(self, x, y, kernel):
        """Emprical maximum mean discrepancy. The lower the result
           the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        if kernel == "multiscale":

            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a ** 2 * (a ** 2 + dxx) ** -1
                YY += a ** 2 * (a ** 2 + dyy) ** -1
                XY += a ** 2 * (a ** 2 + dxy) ** -1

        if kernel == "rbf":

            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)

    def calcMMD(self):

        m = 25  # sample size
        self.hist_1 = torch.from_numpy(self.hist_1)
        self.hist_2 = torch.from_numpy(self.hist_2)
        x = self.hist_1[torch.randperm(len(self.hist_1))[:m]]
        y = self.hist_2[torch.randperm(len(self.hist_2))[:m]]

        result = self.MMD(x, y, kernel='multiscale')

        print(f"MMD result of X and Y is {result.item()}")


stats = StatisticalDetection('D:\\Coding\\PyCharmProjects\\ComputerSecurityProject\\Adversaries\\Images\\test\\*.jpg',
                     'D:\\Coding\\PyCharmProjects\\ComputerSecurityProject\\val2014\\test\\*.jpg')
stats.findDistributions()
stats.calcMMD()
