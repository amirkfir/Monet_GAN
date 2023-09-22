import tensorflow as tf

# pytorch imports
import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(nn.Conv2d(3,16,16,2,1),
                                   nn.LeakyReLU(0.2,inplace=True),
                                   nn.Dropout2d(0.25),
                                   nn.Conv2d(16, 32,16,2,1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout2d(0.25),
                                   nn.Flatten(),
                                   nn.Linear(96800,1),
                                   nn.Softmax())
    def forward(self,img,targets):
        # img_ = tf.convert_to_tensor(torch.swapdims(img,1,3))
        out = self.model(img)
        loss = nn.functional.binary_cross_entropy(out,targets)
        return loss

class Generator(nn.Module):
    def __init__(self,input_length: int):
        super(Generator,self).__init__()
        self.model = nn.Sequential(nn.Linear(input_length,32),
                                   nn.Linear(32,256),
                                   nn.Linear(256,256*256*3),
                                   nn.Unflatten(1,(3,256,256)))

    def forward(self,img):
        out = self.model(img)
        return out

# def get_discriminator (batch_size, image_shape):
#     input_shape = [batch_size] + image_shape
#
#     Discriminator = tf.keras.Sequential(name = "Discriminator")
#     Discriminator.add(layers.Input(image_shape))
#     Discriminator.add(layers.Conv2D(filters = 4, ,stride = (2,2), kernel_size=(32,32), padding = 'same'))
#     # Discriminator.add(layers.MaxPool2D((2,2))
#     Discriminator.add(layers.Conv2D(filters=3, kernel_size=(32, 32), padding='same'))
#     # Discriminator.add(layers.Conv2D(filters=8, kernel_size=(16, 16), padding='same'))
#     # Discriminator.add(layers.Conv2D(filters=4, kernel_size=(8, 8), padding='same'))
#     # Discriminator.add(layers.MaxPool2D((2,2))
#     Discriminator.add(layers.Flatten())
#     # Discriminator.add(layers.Dense(1024))
#     Discriminator.add(layers.Dense(256))
#     Discriminator.add(layers.Dense(32))
#     Discriminator.add(layers.Dense(1))
#     Discriminator.summary()
#     return Discriminator

# def get_generator (batch_size, image_shape,noise_shape):
#     input_shape = [batch_size] + image_shape
#
#     Generator = tf.keras.Sequential(name = "Generator")
#     Generator.add(layers.Input(noise_shape))
#     Generator.add(layers.Dense(32))
#     Generator.add(layers.Dense(256))
#     # Generator.add(layers.Dense(1024))
#     Generator.add(layers.Dense(256*256*3))
#     Generator.add(layers.Reshape(image_shape))
#     Generator.summary()
#     return Generator

# def discriminator_loss(real_output,fake_output):
#     #cross-entropy loss: -sum(p(x)log(q(x))
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
#     real_loss = cross_entropy(real_output,tf.ones_like(real_output))
#     fake_loss = cross_entropy(fake_output,tf.zeros_like(fake_output))
#     return real_loss + fake_loss
#
# def generator_loss(fake_output):
#     #cross-entropy loss: -sum(p(x)log(q(x))
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     return cross_entropy(fake_output,tf.ones_like(fake_output))
#
# @tf.function
# def train_step(images,generator,discriminator,BATCH_SIZE,noise_dim,generator_optimizer,discriminator_optimizer):
#
#     noise = tf.random.normal([BATCH_SIZE,noise_dim])
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)
#         real_out = discriminator(images,training = True)
#         fake_out = discriminator(generated_images, training=True)
#
#         gen_loss = generator_loss(fake_out)
#         disc_loss = discriminator_loss(real_out,fake_out)
#
#         gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#         gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#
#
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
