import numpy as np
import torch.cuda

import Load_Data as LD
import matplotlib.pyplot as plt
import model_config
import time
import train
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader




IMAGE_SIZE = [256, 256,3]
BATCH_SIZE = 25
NOISE_DIM = 20
EPOCH_NUM=100

torch.cuda.is_available()
GCS_PATH = r"/home/amir/PycharmProjects/Monet_GAN/gan-getting-started"

transforms_ = [
    transforms.Resize(int(IMAGE_SIZE[0]*1.12), Image.BICUBIC),
    transforms.RandomCrop((IMAGE_SIZE[0], IMAGE_SIZE[1])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
# MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
# print('Monet TFRecord Files:', len(MONET_FILENAMES))

# PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))
# print('Photo TFRecord Files:', len(PHOTO_FILENAMES))
dataloader = DataLoader(
    LD.ImageDataset(GCS_PATH, transforms_=transforms_, unaligned=True),
    batch_size=BATCH_SIZE, # 1
    shuffle=True,
    num_workers=3 # 3
)

val_dataloader = DataLoader(
    LD.ImageDataset(GCS_PATH, transforms_=transforms_, unaligned=True, mode='test'),
    batch_size=5,
    shuffle=True,
    num_workers=3
)
#
# imgs = next(iter(val_dataloader))
# plt.imshow(np.swapaxes(imgs['A'][0],0,2))
# plt.axis('off')
# plt.show()





train.train(dataloader, BATCH_SIZE,EPOCH_NUM)
# example_monet = next(iter(monet_ds))
# example_photo = next(iter(photo_ds))

# plt.subplot(121)
# plt.title('Photo')
# plt.imshow(example_photo[0] * 0.5 + 0.5)
# #
# plt.subplot(122)
# plt.title('Monet')
# plt.imshow(example_monet[0] * 0.5 + 0.5)

# plt.show()
# Discriminator = model_config.get_discriminator(BATCH_SIZE,IMAGE_SIZE)
# Generator = model_config.get_generator(BATCH_SIZE,IMAGE_SIZE,NOISE_DIM)







