# import tensorflow as tf
import keras.layers as layers

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import model_config
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

def train(batch_dataset, batch_size: int=16,epochs: int = 500):
    noise_length = 20
    generator = model_config.Generator(noise_length).cuda()
    discriminator = model_config.Discriminator().cuda()
    # summary(discriminator, [(3, 256, 256),(1)],)

    generator_opt = torch.optim.Adam(generator.parameters(),lr=0.001)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(),lr=0.001)
    # loss = nn.BCELoss()

    real_labels = torch.ones(batch_size,1).cuda()
    fake_labels = torch.zeros(batch_size, 1).cuda()

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        generator_opt.zero_grad()

        for i, batch in enumerate(tqdm(batch_dataset)):
            image_batch = batch['A'].cuda()

            noise = torch.randn(batch_size,noise_length)
            noise = noise.cuda()
            generator.eval()
            with torch.no_grad():
                generated_batch = generator(noise)

            discriminator.train()

            d_loss = discriminator(generated_batch,fake_labels) + discriminator(image_batch,real_labels)

            discriminator_opt.zero_grad()
            d_loss.backward()
            discriminator_opt.step()


            generator.train()
            noise = torch.randn(batch_size,noise_length).cuda()
            generated_batch = generator(noise)

            discriminator.eval()
            g_loss = discriminator(generated_batch,fake_labels)

            generator_opt.zero_grad()
            g_loss.backward()
            generator_opt.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())



        print(epoch, np.mean(d_losses), np.mean(g_losses))
        image = generated_batch[0].cpu().detach().numpy()
        plt.imshow(np.swapaxes(abs(image)/image.max(), 0, 2))








