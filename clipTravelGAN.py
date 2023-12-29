# from https://www.kaggle.com/code/unfriendlyai/cliptravelgan-monet-article-implementation


BATCH_SIZE = 128  # change in siamis loss too

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import TFCLIPVisionModel


def get_scale_layer():
    mean = np.array([0.48145466, 0.4578275, 0.40821073]) * 2 - 1
    std = np.array([0.26862954, 0.26130258, 0.27577711]) * 2
    scaling_layer = keras.layers.Lambda(lambda x: (tf.cast(x, tf.float32) - mean) / std)

    return scaling_layer


# clip model and preprocessing according to clip config
def get_clip_model():
    layer_scaling = get_scale_layer()
    layer_permute = tf.keras.layers.Permute((3, 1, 2))
    backbone = TFCLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

    inp = tf.keras.layers.Input(shape=[256, 256, 3])  # [B, C, H, W]
    x = inp[:, 16:240, 16:240, :]
    x = layer_scaling(x)
    x = layer_permute(x)

    output = backbone({'pixel_values': x}).pooler_output

    return tf.keras.Model(inputs=[inp], outputs=[output])


OUTPUT_CHANNELS = 3


def down_sample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    layer = keras.Sequential()
    layer.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    layer.add(layers.LeakyReLU())

    return layer


def up_sample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    layer = keras.Sequential()
    layer.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer,
                                     use_bias=False))
    layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        layer.add(layers.Dropout(0.5))

    layer.add(layers.ReLU())

    return layer


def Generator():
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)  # 256*256*64

    x1 = tf.keras.layers.MaxPooling2D(padding='same')(x)  # 128*128*64

    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer, activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer, activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # 128*128*128

    x2 = tf.keras.layers.MaxPooling2D(padding='same')(x1)  # 64*64*128

    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=initializer, activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=initializer, activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # 64*64*256

    x3 = tf.keras.layers.MaxPooling2D(padding='same')(x2)  # 32*32*256

    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer, activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer, activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # 32*32*512

    x4 = tf.keras.layers.MaxPooling2D(padding='same')(x3)  # 16*16*512

    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer=initializer, activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer=initializer, activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # 16*16*1024

    x14 = tf.keras.layers.MaxPooling2D(padding='same')(x4)  # 16*16*512

    x14 = tf.keras.layers.Conv2D(2048, 3, padding='same', kernel_initializer=initializer, activation='relu')(x14)
    x14 = tf.keras.layers.BatchNormalization()(x14)
    x14 = tf.keras.layers.Conv2D(2048, 3, padding='same', kernel_initializer=initializer, activation='relu')(x14)
    x14 = tf.keras.layers.BatchNormalization()(x14)  # 8*8*2048

    x15 = tf.keras.layers.Conv2DTranspose(1024, 2, strides=2, padding='same', kernel_initializer=initializer,
                                          activation='relu')(x14)
    x15 = tf.keras.layers.BatchNormalization()(x15)  # 32*32*512

    x16 = tf.concat([x4, x15], axis=-1)  # 32*32*1024

    x16 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer=initializer, activation='relu')(x16)
    x16 = tf.keras.layers.BatchNormalization()(x16)
    x16 = tf.keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer=initializer, activation='relu')(x16)
    x16 = tf.keras.layers.BatchNormalization()(x16)  # 32*32*512

    x5 = tf.keras.layers.Conv2DTranspose(1024, 2, strides=2, padding='same', kernel_initializer=initializer,
                                         activation='relu')(x16)
    x5 = tf.keras.layers.BatchNormalization()(x5)  # 32*32*512

    x6 = tf.concat([x3, x5], axis=-1)  # 32*32*1024

    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer, activation='relu')(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=initializer, activation='relu')(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)  # 32*32*512

    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same', kernel_initializer=initializer,
                                         activation='relu')(x6)
    x7 = tf.keras.layers.BatchNormalization()(x7)  # 64*64*256

    x8 = tf.concat([x2, x7], axis=-1)  # 64*64*512

    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=initializer, activation='relu')(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)
    x8 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=initializer, activation='relu')(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)  # 64*64*256

    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', kernel_initializer=initializer,
                                         activation='relu')(x8)
    x9 = tf.keras.layers.BatchNormalization()(x9)  # 128*128*128

    x10 = tf.concat([x1, x9], axis=-1)  # 128*128*256

    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer, activation='relu')(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)
    x10 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=initializer, activation='relu')(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)  # 128*128*128

    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same', kernel_initializer=initializer,
                                          activation='relu')(x10)
    x11 = tf.keras.layers.BatchNormalization()(x11)  # 256*256*64

    x12 = tf.concat([x, x11], axis=-1)  # 256*256*128

    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer, activation='relu')(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)
    x12 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=initializer, activation='relu')(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)  # 256*256*64

    outputs = tf.keras.layers.Conv2D(3, 1, kernel_initializer=initializer, activation='tanh')(x12)  # 256*256*3

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')
    x = inp

    down1 = down_sample(64, 4, False)(x)  # (size, 128, 128, 64)
    down2 = down_sample(128, 4)(down1)  # (size, 64, 64, 128)
    down3 = down_sample(256, 4)(down2)  # (size, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)  # (size, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(
        zero_pad1)  # (size, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)
    leaky_relu = layers.LeakyReLU()(norm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (size, 33, 33, 512)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (size, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


with strategy.scope():
    monet_generator = Generator()  # transforms photos to Monet-esque paintings
    monet_discriminator = Discriminator()  # differentiates real Monet paintings and generated Monet paintings
    clip_net = get_clip_model()
    clip_net.trainable = False


class CLIPTraVeLGan(keras.Model):
    def __init__(
            self,
            monet_generator,
            monet_discriminator,
            siames_net,
            lambda_id=0.00001  # balance between adversarial loss and clip loss
    ):
        super(CLIPTraVeLGan, self).__init__()
        self.m_gen = monet_generator
        self.m_disc = monet_discriminator
        self.siames_net = siames_net
        self.lambda_id = lambda_id

    def compile(
            self,
            m_gen_optimizer,
            m_disc_optimizer,
            gen_loss_fn,
            disc_loss_fn,
            siames_loss_fn
    ):
        super(CLIPTraVeLGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.siames_loss_fn = siames_loss_fn

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        batch_size = tf.shape(real_monet)[0]
        semantic_real = self.siames_net(real_photo, training=False)  # CLIP embedding of real
        with tf.GradientTape(persistent=True) as tape:
            fake_monet = self.m_gen(real_photo, training=True)
            semantic_fake = self.siames_net(fake_monet, training=False)  # CLIP embedding of fake

            ################## My code #####################

            both_monet = tf.concat([real_monet, fake_monet], axis=0)

            aug_monet = aug_fn(both_monet)

            aug_real_monet = aug_monet[:batch_size]
            aug_fake_monet = aug_monet[batch_size:]

            ################ End of my code #################

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(aug_real_monet, training=True)  # aug_real_monet

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(aug_fake_monet, training=True)  # aug_fake_monet

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)

            # travel loss on CLIP embeddings

            monet_travel_loss = self.siames_loss_fn(semantic_real, semantic_fake)

            # evaluates generator loss
            total_monet_gen_loss = self.gen_loss_fn(disc_fake_monet) + monet_travel_loss * self.lambda_id

        # Calculate the gradients for generator and discriminator
        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)

        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)

        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        return {
            "disc_real_monet": disc_real_monet,
            "disc_fake_monet": disc_fake_monet,
            "monet_disc_loss": monet_disc_loss,
            "monet_travel_loss": monet_travel_loss,
        }


# Loss functions

with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
            tf.ones_like(real), real)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
            tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

with strategy.scope():
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
            tf.ones_like(generated), generated)

with strategy.scope():
    def siames_loss(s1_x, s1_g):
        orders = np.array([list(range(i, 128)) + list(range(i)) for i in range(1, 128)])  # change 128 to batch_size
        orders = tf.constant(orders)

        orders2 = np.array([list(range(0, 128)) for i in range(1, 128)])  # change 128 to batch_size
        orders2 = tf.constant(orders2)

        dists_within_x1 = tf.gather(s1_x, orders2) - tf.gather(s1_x, orders)
        dists_within_g1 = tf.gather(s1_g, orders2) - tf.gather(s1_g, orders)

        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
        losses_travel_1 = tf.reduce_sum(cosine_loss(dists_within_x1, dists_within_g1) + 1)

        return losses_travel_1

# Create a model

with strategy.scope():
    gan_model = CLIPTraVeLGan(monet_generator, monet_discriminator, clip_net)

    monet_generator.built = True
    monet_discriminator.built = True

with strategy.scope():
    for (lr, stg, ep) in [(2e-4, 7, 1), (1e-4, 5, 1), (3e-5, 2, 1)]:
        print(f"Learnning rate = {lr}")
        monet_generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        monet_discriminator_optimizer = tf.keras.optimizers.Adam(lr * 2, beta_1=0.5)

        gan_model.compile(
            m_gen_optimizer=monet_generator_optimizer,
            m_disc_optimizer=monet_discriminator_optimizer,
            gen_loss_fn=generator_loss,
            disc_loss_fn=discriminator_loss,
            siames_loss_fn=siames_loss,
        )

        for stage in range(1, stg + 1):
            print("Stage = ", stage)
            hist = gan_model.fit(gan_ds, steps_per_epoch=1400, epochs=ep).history
            disc_m_loss.append(hist["monet_disc_loss"][0])
            #             cur_fid = FID(fid_photo_ds, monet_generator)
            #             fids.append(cur_fid)
            #             print("After stage #{} FID = {} \n".format(stage, cur_fid))

            #             if cur_fid<best_fid:
            #                         print(f"{cur_fid} is better than previous bestFID {best_fid} \n")
            #                         best_fid=cur_fid
            #                         monet_generator.save_weights("monet_generator.h5")
            #                         monet_discriminator.save_weights("monet_discriminator.h5")

            if stage == stg:
                ds_iter = iter(fid_photo_ds)
                example_sample = next(ds_iter)
                generated_sample = monet_generator.predict(example_sample)
                for n_sample in range(8):
                    f = plt.figure(figsize=(32, 32))
                    plt.subplot(121)
                    plt.title('Input image')
                    plt.imshow(example_sample[n_sample] * 0.5 + 0.5)
                    plt.axis('off')
                    plt.subplot(122)
                    plt.title('Generated image')
                    plt.imshow(generated_sample[n_sample] * 0.5 + 0.5)
                    plt.axis('off')
                    plt.show()

#         monet_generator.load_weights("monet_generator.h5")
#         monet_discriminator.load_weights("monet_discriminator.h5")
