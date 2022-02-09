import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math

logistic_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class DCGAN:
    def __init__(self, foldername, imgres, noise_dim, batch_size=64):
        """
        Parameters
        ----------
        foldername: string
            Path to folder containing training data
        imgres: int
            Resolution of the square images used in the training set
        noise_dim: int
            Dimension of the noise to feed the model
        batch_size: int
            Size of each batch to use during training
        """
        self.foldername = foldername
        self.imgres = imgres
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        datagen = ImageDataGenerator(preprocessing_function = lambda x: (x-255/2)/(255/2))
        self.dataset = datagen.flow_from_directory(foldername, class_mode=None, batch_size=batch_size, 
                                                    shuffle=True, target_size=(imgres, imgres), 
                                                    color_mode="grayscale")
        self.make_generator()
        self.make_discriminator()
        self.setup_checkpoints()

    def tanh(x):
        return Math.tanh
    
    def make_generator(self):
        """
        Create the CNN for the generator network
        """
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.generator = tf.keras.Sequential()
        
        ## TODO: Fill this in to make the generator CNN
        self.generator.add(layers.Dense(units=((self.imgres//4)**2 * 32), use_bias=False, input_shape=(self.noise_dim,)))
        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())
        self.generator.add(layers.Reshape((self.imgres//4, self.imgres//4, 32)))
        self.generator.add(layers.Conv2DTranspose(filters=32, use_bias=False, kernel_size=(5, 5), strides=(2,2), padding="same" ))
        self.generator.add(layers.BatchNormalization())
        self.generator.add(layers.LeakyReLU())
        self.generator.add(layers.Conv2DTranspose(filters=1, use_bias=False, kernel_size=(5, 5), strides=(2,2), activation="tanh", padding="same"))

        print("========================================\nGENERATOR\n\n")
        self.generator.summary()

    def make_discriminator(self):
        """
        Create the CNN for the discriminator network
        """
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator = tf.keras.Sequential()
        
        ## TODO: Fill this in to create the discriminator CNN
        self.discriminator.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2), input_shape=(self.imgres, self.imgres, 1)))
        self.discriminator.add(layers.LeakyReLU())
        self.discriminator.add(layers.Dropout(0.3))
        self.discriminator.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2)))
        self.discriminator.add(layers.LeakyReLU())
        self.discriminator.add(layers.Dropout(0.3))
        self.discriminator.add(layers.Flatten())
        self.discriminator.add(layers.Dense(units=1))

        print("========================================\nDISCRIMINATOR\n\n")
        self.discriminator.summary()
    
    def setup_checkpoints(self):
        """
        Setup a checkpoints object to save the model as it progresses
        """
        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

    def generator_loss(self, fake_estimates):
        """
        Return the generator loss, which is the cross-entropy between 
        what the discriminator network estimated the fake images were and
        the "ground truth" tricky label of a 1 for each fake image
        
        Parameters
        ----------
        fake_estimates: tensor
            A 1D tensor of the logistic function estimate that the discriminator
            gave to a batch of fake images
        
        Returns
        -------
        float:
            Logistic loss (cross-entropy) between a vector of all 1's and the 

        """
        return logistic_loss(tf.ones_like(fake_estimates), fake_estimates)
    
    def discriminator_loss(self, real_estimates, fake_estimates):
        """
        Return the discriminator loss based on the fact that it should classify
        true images as 1 and fake images as 0

        Parameters
        ----------
        real_estimates: tensor
            A 1D tensor of the logistic function estimate that the discriminator
            gave to a batch of real images
        fake_estimates: tensor
            A 1D tensor of the logistic function estimate that the discriminator
            gave to a batch of fake images
        """
        loss = 0
        ## TODO: Fill this in
        loss = logistic_loss(tf.ones_like(real_estimates), real_estimates) + logistic_loss(tf.zeros_like(fake_estimates), fake_estimates)
        return loss

    @tf.function
    def train_step(self, images):
        """
        Do a training step on a batch of images

        Parameters
        ----------
        images: tf.tensor(N, imgres, imgres, 1)
            A batch of images on which to train
        """
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_estimates = self.discriminator(images, training=True)
            fake_estimates = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_estimates)
            disc_loss = self.discriminator_loss(real_estimates, fake_estimates)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def train(self, epochs):
        """
        Train the network for some number of epochs (iterations), and show progress
        by plotting the outputs of the same random inputs at the end of every epoch

        Parameters
        ----------
        epochs: int
            Number of training iterations to do
        """
        # Same inputs that will be visualized over time
        KxK = 16
        seed = tf.random.normal([KxK, self.noise_dim]) 

        epoch = 0
        batch = 0
        batches_per_epoch = len(self.dataset)
        tic = time.time()
        all_gen_loss = []
        all_disc_loss = []
        while epoch < epochs:
            image_batch = self.dataset.next()
            gen_loss, disc_loss = self.train_step(image_batch)
            all_gen_loss.append(gen_loss)
            all_disc_loss.append(disc_loss)
            batch += 1
            if batch%20 == 0:
                print("Epoch {} batch {} of {}".format(epoch, batch, batches_per_epoch))
            if batch%batches_per_epoch == 0:
                plt.figure(figsize=(12, 6))
                xepoch = np.arange(len(all_gen_loss))/batches_per_epoch
                plt.plot(xepoch, all_gen_loss)
                plt.plot(xepoch, all_disc_loss)
                plt.xticks(np.arange(epoch+2))
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Losses")
                plt.legend(["Generator Loss", "Discriminator Loss"])
                
                epoch += 1
                batch = 0
                self.plot_image_grid(seed, KxK)
                self.train_step(image_batch)
                # Save the model every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.checkpoint.save(file_prefix = self.checkpoint_prefix)

                print ('Time for epoch {} is {} sec'.format(epoch, time.time()-tic))
                tic = time.time()
        return all_gen_loss, all_disc_loss

    def plot_image_grid(self, seed, N):
        """
        Plot a grid of images

        Parameters
        ----------
        seed: tf.tensor([KxK, noise_dim])
            An array of different draws of noise for generating
            KxK random images using the network
        N: int
            KxK
        """
        # Training is set to false so layers run in inference mode (batchnorm)
        predictions = self.generator(seed, training=False)
        plt.figure(figsize=(20, 20))
        K = int(np.ceil(np.sqrt(N)))
        for i in range(predictions.shape[0]):
            plt.subplot(K, K, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
        plt.show()
    
    def generate_random_image(self):
        """
        Plot a single random image generated by the network.  Result will be different each time
        """
        noise = tf.random.normal([1, self.noise_dim])
        generated_image = self.generator(noise, training=False)
        plt.imshow(generated_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray', vmin=0, vmax=255)