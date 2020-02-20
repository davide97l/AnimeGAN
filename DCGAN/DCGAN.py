import os
import time
import numpy as np
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import glob
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from collections import deque
from keras.models import load_model
import sys
import argparse


def generator(noise_shape=(1, 1, 100)):
    noise_shape = noise_shape

    kernel_init = 'glorot_uniform'

    gen_input = Input(shape=noise_shape)
    generator = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(1, 1), padding="valid",
                                kernel_initializer=kernel_init)(gen_input)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    # output (?, 4, 4, 512)

    generator = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    # output (?, 8, 8, 256)

    generator = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    # output (?, 16, 16, 128)

    generator = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    # output (?, 32, 32, 64)

    generator = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                       kernel_initializer=kernel_init)(generator)
    generator = BatchNormalization(momentum=0.5)(generator)
    generator = LeakyReLU(0.2)(generator)
    # output (?, 32, 32, 64)

    generator = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                kernel_initializer=kernel_init)(generator)
    generator = Activation('tanh')(generator)
    # output (?, 64, 64, 3)

    gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(input=gen_input, output=generator)
    generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])

    return generator_model


def discriminator(image_shape=(64, 64, 3)):
    image_shape = image_shape

    kernel_init = 'glorot_uniform'

    dis_input = Input(shape=image_shape)
    discriminator = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                           kernel_initializer=kernel_init)(dis_input)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)
    discriminator = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same",
                           kernel_initializer=kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum=0.5)(discriminator)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Flatten()(discriminator)

    discriminator = Dense(1)(discriminator)
    discriminator = Activation('sigmoid')(discriminator)

    dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(input=dis_input, output=discriminator)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=dis_opt, metrics=['accuracy'])

    return discriminator_model


def norm_img(img):
    # image normalization to keep values between -1 and 1 for stability
    img = (img / 127.5) - 1
    return img


def denorm_img(img):
    # inverse operation of imahe normalization
    img = (img + 1) * 127.5
    return img.astype(np.uint8)


def sample_from_dataset(batch_size, image_shape, data_dir=None):
    # take a sample of batch_size random images from the dataset
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    # get all the name files in the directory
    all_data_dirlist = list(glob.glob(data_dir + "*.png"))
    sample_imgs_paths = np.random.choice(all_data_dirlist, batch_size)
    for index, img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB')
        image = np.asarray(image)
        image = norm_img(image)
        sample[index] = image
    return sample


def gen_noise(batch_size, noise_shape):
    # input noise for the generator should follow a probability distribution, like in this case normal distribution
    return np.random.normal(0, 1, size=(batch_size,) + noise_shape)


def generate_images(generator, save_dir, batch_size, noise_shape, grid_size=4):
    # generate sample images using only the generator
    noise = gen_noise(batch_size, noise_shape)
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(grid_size, grid_size))
    gs1 = gridspec.GridSpec(grid_size, grid_size)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0], grid_size**2, replace=False)
    for i in range(grid_size**2):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :, :, :]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir + str(time.time()) + "_generated_images.png", bbox_inches='tight', pad_inches=0)
    plt.show()


def display_real_images(save_dir, real_data_X, grid_size=4):
    print("Displaying real images")
    plt.figure(figsize=(grid_size, grid_size))
    gs1 = gridspec.GridSpec(grid_size, grid_size)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(real_data_X.shape[0], grid_size**2, replace=False)
    for i in range(grid_size**2):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = real_data_X[rand_index, :, :, :]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir + str(time.time()) + "_sampled_images.png", bbox_inches='tight', pad_inches=0)
    plt.show()


def save_img_batch(img_batch, save_dir):
    # save the img_batch images int the save_dir directory
    plt.figure(figsize=(4, 4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0], 16, replace=False)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :, :, :]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    """
    example: python DCGAN.py -n AnimeGAN -d anime_dataset
    example: python DCGAN.py -n AnimeGAN -d anime_dataset -t 20000
    example: python DCGAN.py -n AnimeGAN -i 19999 -g 0 -x 4
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", type=str, required=True,
                    help="project name, used to generate the relative folders")
    ap.add_argument("-d", "--dataset_dir", type=str, required=True,
                    help="directory where the dataset is located")
    ap.add_argument("-t", "--training_steps", type=int, default=10000,
                    help="number of training steps")
    ap.add_argument("-b", "--batch_size", type=int, default=64,
                    help="batch size")
    ap.add_argument("-i", "--initial_step", type=int, default=0,
                    help="skips the first initial_step steps and load model relative to initial_step")
    ap.add_argument("-sm", "--steps_to_save", type=int, default=1000,
                    help="number of training steps between save model")
    ap.add_argument("-ss", "--steps_to_sample", type=int, default=200,
                    help="number of training steps between save image samples")
    ap.add_argument("-m", "--mode", type=int, default=0,
                    help="0 training, 1 generation, 2 display real samples")
    ap.add_argument("-w", "--window_size", type=int, default=4,
                    help="when generating images this is the size of a window containing w*w images")
    args = vars(ap.parse_args())

    noise_shape = (1, 1, 100)
    num_steps = args["training_steps"]
    save_sample_steps = args["steps_to_sample"]
    batch_size = args["batch_size"]
    save_model_steps = args["steps_to_save"]
    image_shape = (64, 64, 3)
    project_name = args["name"]
    data_dir = args["dataset_dir"] + "/"
    initial_step = args["initial_step"]  # if load_model_dir is not None then load the model at the step initial_step
    mode = args["mode"]
    window_size = args["window_size"]

    img_save_dir = "results/" + project_name + "/"
    log_dir = "logs/" + project_name + "/"
    save_model_dir = "model/" + project_name + "/"

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    discriminator = discriminator(image_shape)
    generator = generator(noise_shape)
    if initial_step > 0:
        generator = load_model(save_model_dir + str(initial_step) + "_generator.hdf5")
        discriminator = load_model(save_model_dir + str(initial_step) + "_discriminator.hdf5")

    discriminator.trainable = False

    opt = Adam(lr=0.00015, beta_1=0.5)
    gen_inp = Input(shape=noise_shape)
    GAN_inp = generator(gen_inp)
    GAN_opt = discriminator(GAN_inp)
    gan = Model(input=gen_inp, output=GAN_opt)
    gan.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    avg_disc_fake_loss = deque([0], maxlen=250)
    avg_disc_real_loss = deque([0], maxlen=250)
    avg_GAN_loss = deque([0], maxlen=250)

    noise = gen_noise(batch_size, noise_shape)
    fake_data_X = generator.predict(noise)

    if mode == 1:
        generate_images(generator, img_save_dir, batch_size, noise_shape, window_size)
        sys.exit()

    if mode == 2:
        real_data_X = sample_from_dataset(batch_size, image_shape, data_dir=data_dir)
        display_real_images(img_save_dir, real_data_X, window_size)
        sys.exit()

    # start training
    for step in range(num_steps):
        if step <= initial_step:
            continue

        step_begin_time = time.time()

        # make some samples from the real data
        real_data_X = sample_from_dataset(batch_size, image_shape, data_dir=data_dir)

        # generate noise
        noise = gen_noise(batch_size, noise_shape)

        # generate some fake samples
        fake_data_X = generator.predict(noise)

        # save generated samples
        if (step % save_sample_steps) == 0:
            step_num = str(step).zfill(6)
            save_img_batch(fake_data_X, img_save_dir + step_num + ".png")

        # concatenate real and fake data samples
        data_X = np.concatenate([real_data_X, fake_data_X])

        # add noise to the real label inputs (which will still be close to 1)
        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2

        # add noise to the fake label inputs (which will still be close to 0)
        fake_data_Y = np.random.random_sample(batch_size) * 0.2

        # concatenate real and fake labels
        data_Y = np.concatenate((real_data_Y, fake_data_Y))

        generator.trainable = False
        discriminator.trainable = True
        # training the discriminator first on the real data, then on the fake one
        dis_metrics_real = discriminator.train_on_batch(real_data_X, real_data_Y)
        dis_metrics_fake = discriminator.train_on_batch(fake_data_X, fake_data_Y)

        # append the respective loss
        avg_disc_fake_loss.append(dis_metrics_fake[0])
        avg_disc_real_loss.append(dis_metrics_real[0])

        # not train generator
        discriminator.trainable = False
        generator.trainable = True

        # generate samples for GAN
        GAN_X = gen_noise(batch_size, noise_shape)
        GAN_Y = real_data_Y

        # we have to make the discriminator take the wrong decision so positive label for negative sample
        gan_metrics = gan.train_on_batch(GAN_X, GAN_Y)

        # save log of current step
        text_file = open(log_dir + "/training_log.txt", "a")
        log_line = "Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (step, dis_metrics_real[0],
                                                                                  dis_metrics_fake[0], gan_metrics[0])
        text_file.write(log_line)
        text_file.close()
        print(log_line)
        avg_GAN_loss.append(gan_metrics[0])

        # take step time
        end_time = time.time()
        diff_time = int(end_time - step_begin_time)
        print("Time took: %s secs." % diff_time)

        # save model
        if save_model_steps is not None and (step % save_model_steps) == 0:
            print("-----------------------------------------------------------------")
            print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))
            print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))
            print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
            print("-----------------------------------------------------------------")
            discriminator.trainable = True
            generator.trainable = True
            generator.save(save_model_dir + str(step) + "_generator.hdf5")
            discriminator.save(save_model_dir + str(step) + "_discriminator.hdf5")
