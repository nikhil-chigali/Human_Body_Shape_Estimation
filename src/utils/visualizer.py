import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from consts import MEAN, STD


def visualize_batch(imgs):
    imgs = make_grid(imgs)
    imgs = imgs.cpu().numpy()
    mean = np.array(MEAN).reshape((3, 1, 1))
    std = np.array(STD).reshape((3, 1, 1))
    imgs = (imgs * std + mean).transpose(1, 2, 0)
    plt.imshow(imgs)
    plt.show()
