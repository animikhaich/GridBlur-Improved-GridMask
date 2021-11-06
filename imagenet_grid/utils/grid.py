import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math, cv2
import matplotlib.pyplot as plt


class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.0):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob
        
        self.lower_offset = 2.117904
        self.upper_offset = 2.64

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img

        img = img.cpu().numpy()

        # Channel First --> Channel Last
        temp_img = img.transpose(1, 2, 0)
        
        h = temp_img.shape[0]
        w = temp_img.shape[1]

        # Normalize Image
        temp_img = temp_img + self.lower_offset
        temp_img = temp_img / (self.lower_offset + self.upper_offset)
        temp_img[temp_img > 1.0] = 1.0
        temp_img[temp_img < 0.0] = 0.0


        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        r = np.random.randint(self.rotate)

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        # Blur Image and Resize
        ksize = np.random.randint(5, hh // 10)
        temp_img = np.uint8(temp_img * 255)
        resized = cv2.resize(temp_img, (hh, hh), cv2.INTER_AREA)
        blurred = cv2.blur(resized, (ksize, ksize))

        # Rotate and Crop the blurred image
        blurred = Image.fromarray(np.uint8(blurred))
        blurred = blurred.rotate(r)
        blurred = np.asarray(blurred)
        blurred = blurred[
            (hh - h) // 2 : (hh - h) // 2 + h, (hh - w) // 2 : (hh - w) // 2 + w
        ]

        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0

        # Rotate and Crop mask
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (hh - w) // 2 : (hh - w) // 2 + w
        ]
        mask = mask.astype(np.float)

        if self.mode == 0:
            mask = 1 - mask

        # Convert Mask to 3 channels
        mask = np.dstack([mask] * 3)

        # Unify dypes and generate inverse mask
        img = np.uint8(img * 255)
        mask = np.uint8(mask * 255)
        blurred = np.uint8(blurred)
        mask_inv = 255 - mask

        # Generate complementary foreground and backgreound images
        blurred_masked = cv2.bitwise_and(blurred, mask)
        image_masked = cv2.bitwise_and(temp_img, mask_inv)

        # Create the final image
        final_image = cv2.bitwise_or(blurred_masked, image_masked)

        # Channel last --> Channel first
        final_image = final_image.transpose(2, 0, 1)

        # Reverse Normalization
        final_image = final_image / 255
        temp_img = temp_img * (self.lower_offset + self.upper_offset)
        temp_img = temp_img - self.lower_offset
        
        # Convert back to torch tensor
        final_image = torch.from_numpy(final_image).float().cuda()

        return final_image


class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.0):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x
        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n, c, h, w)
        return y

