import os
import glob
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StatisticalDetection:

    def __init__(self, folder1path, folder2path):
        self.adv_hist = np.zeros([256, 1], dtype=float)
        self.clean_hist = np.zeros([256, 1], dtype=float)
        # Read all the images in as grayscale
        filenames1 = glob.glob(folder1path)
        filenames1.sort()
        filenames2 = glob.glob(folder2path)
        filenames2.sort()
        self.adv_images = [cv2.imread(imgfile, 0) for imgfile in filenames1]
        self.clean_images = [cv2.imread(imgfile, 0) for imgfile in filenames2]
        # Init ORB object
        self.orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

    def findDistributions(self):

        # Perform ORB feature extraction (adv_images and clean_images are same length)
        for i in range(len(self.adv_images)):
            # Adv image first
            keypoints = self.orb.detect(self.adv_images[i], None)
            keypoints, _ = self.orb.compute(self.adv_images[i], keypoints)
            kp_img = cv2.drawKeypoints(self.adv_images[i], keypoints, None, color=(0, 255, 0), flags=0)
            hist = cv2.calcHist([kp_img], [0], None, [256], [0, 256])
            np.add(self.adv_hist, hist, out=self.adv_hist)
            # Repeat for clean image
            keypoints = self.orb.detect(self.clean_images[i], None)
            keypoints, _ = self.orb.compute(self.clean_images[i], keypoints)
            kp_img = cv2.drawKeypoints(self.clean_images[i], keypoints, None, color=(0, 255, 0), flags=0)
            hist = cv2.calcHist([kp_img], [0], None, [256], [0, 256])
            np.add(self.clean_hist, hist, out=self.clean_hist)

        plt.plot(self.adv_hist)
        plt.show()
        plt.plot(self.clean_hist)
        plt.show()

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

        m = 100  # sample size
        x_mean = torch.mean(torch.from_numpy(self.adv_hist))
        y_mean = torch.mean(torch.from_numpy(self.clean_hist))
        x_cov = 2 * torch.eye(256)  # IMPORTANT: Covariance matrices must be positive definite
        y_cov = 3 * torch.eye(256) - 1

        x = torch.from_numpy(self.adv_hist)
        y = torch.from_numpy(self.clean_hist)

        result = self.MMD(x, y, kernel='multiscale')

        print(f"MMD result of X and Y is {result.item()}")


stats = StatisticalDetection('D:\\Coding\\PyCharmProjects\\ComputerSecurityProject\\Adversaries\\Images\\dev\\*.jpg',
                     'D:\\Coding\\PyCharmProjects\\ComputerSecurityProject\\val2014\\dev\\*.jpg')
stats.findDistributions()
stats.calcMMD()
