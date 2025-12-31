def encoder_model(latent_dim):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    
    x = layers.Conv2D(32, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, 3, strides=2, padding='same')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, 3, strides=2, padding='same')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
     
    latent = layers.Dense(
    latent_dim, 
    name='latent_layer',
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
    bias_initializer=tf.keras.initializers.Zeros())(x)
    return tf.keras.Model(inputs, latent, name='encoder')

def decoder_model(latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(4 * 4 * 256)(latent_inputs) 
    x = layers.Reshape((4, 4, 256))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x) 
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x) 

    return tf.keras.Model(latent_inputs, x, name='decoder')

def discriminator_model(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dropout(0.6)(x) 
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation=None)(x)
    return tf.keras.Model(inputs, output, name='discriminator')

class AAE(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.warmup_epochs = 50
        self.adversarial_ramp_epochs = 150 
        self.max_adversarial_weight = 0.3
        self.reconstruction_weight = 7
        
        self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.train_recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.d_train_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_train_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_recon_loss_tracker = tf.keras.metrics.Mean(name="val_recon_loss")
        self.val_d_train_loss_tracker = tf.keras.metrics.Mean(name="val_d_loss")
        self.val_g_train_loss_tracker = tf.keras.metrics.Mean(name="val_g_loss")
        
        self.bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        
    def call(self, x):
      return self.decoder(self.encoder(x))

    def compile(self, d_optimizer, optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.optimizer = optimizer

    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.val_loss_tracker,
            self.train_recon_loss_tracker,
            self.val_recon_loss_tracker,
            self.d_train_loss_tracker,
            self.val_d_train_loss_tracker,
            self.g_train_loss_tracker,
            self.val_g_train_loss_tracker

        ]

    def get_adversarial_weight(self):
        epoch = tf.cast(self.current_epoch, tf.float32)
        warmup = tf.cast(self.warmup_epochs, tf.float32)
        
        if epoch < warmup:
            return 0.0
        progress = tf.minimum((epoch - warmup) / self.adversarial_ramp_epochs, 1.0)
        smooth_progress = progress ** 3
        return self.max_adversarial_weight * smooth_progress

    def reset_discriminator_if_needed(self, epoch):
        if epoch > self.warmup_epochs and (epoch - self.warmup_epochs) % 30:
            for layer in self.discriminator.layers:
                if hasattr(layer, 'kernel_initializer'):
                    weights = layer.get_weights()
                    weights[0] = layer.kernel_initializer(shape=weights[0].shape)
                    if len(weights) > 1:
                        weights[1] = layer.bias_initializer(shape=weights[1].shape)
                    layer.set_weights(weights)
            return True
        return False


    @tf.function
    def train_step(self, data):
        batch_size = tf.shape(data[0])[0]
        adversarial_weight = self.get_adversarial_weight()
        with tf.GradientTape() as ae_tape:
            z = self.encoder(data[0], training=True)
            z_mean = tf.reduce_mean(z, axis=0)  
            z_mean_norm = tf.norm(z_mean)   
            z_std = tf.math.reduce_std(z, axis=0)
            z_coverage = tf.reduce_mean(tf.cast(tf.abs(z) < 3.0, tf.float32))  
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = rmse(data[0], reconstruction)
            
            z_var = tf.math.reduce_variance(z, axis=0)
            mom_loss =0.2 * (tf.reduce_mean(tf.square(z_mean)) +
                               tf.reduce_mean(tf.abs(z_var - 1.0)))
            
            mean_centering_loss = 0.2 * tf.reduce_mean(tf.abs(tf.reduce_mean(z, axis=0)))

            total_recon_loss = self.reconstruction_weight * reconstruction_loss + mom_loss + mean_centering_loss

        ae_vars = self.encoder.trainable_weights + self.decoder.trainable_weights
        ae_grads = ae_tape.gradient(total_recon_loss, ae_vars)
        ae_grads = [tf.clip_by_norm(g, 1) for g in ae_grads]
        self.optimizer.apply_gradients(zip(ae_grads, ae_vars))                    
        g_loss = 0.0
        if adversarial_weight > 0:
            with tf.GradientTape() as g_tape:
                z = self.encoder(data[0], training=True)
                z_noisy = z + tf.random.normal(tf.shape(z), stddev=0.01)
                g_loss = self.bce_logits(tf.ones((batch_size, 1)), 
                                self.discriminator(z_noisy, training=False))
                total_g_loss = adversarial_weight * g_loss
  
            g_vars = self.encoder.trainable_weights  
            g_grads = g_tape.gradient(total_g_loss, g_vars)
            g_grads = [tf.clip_by_norm(g, 2) for g in g_grads]
            self.optimizer.apply_gradients(zip(g_grads, g_vars))
           
        else: 
            z = self.encoder(data[0], training=False)
            z_noisy = z + tf.random.normal(tf.shape(z), stddev=0.01)
            g_loss = self.bce_logits(tf.ones((batch_size, 1)), 
                            self.discriminator(z_noisy, training=False))
            total_g_loss = adversarial_weight * g_loss
        d_loss = 0.0
        if adversarial_weight > 0 and tf.random.uniform([]) < 0.35: 
            with tf.GradientTape() as d_tape:
                z_real = tf.random.normal((batch_size, self.latent_dim))
                z_fake = tf.stop_gradient(self.encoder(data[0], training=False))
                z_real_noisy = z_real + tf.random.normal(tf.shape(z_real), stddev=0.02)
                z_fake_noisy = z_fake + tf.random.normal(tf.shape(z_fake), stddev=0.02)
                d_real = self.discriminator(z_real_noisy, training=True)
                d_fake = self.discriminator(z_fake_noisy, training=True)
                d_loss_real = self.bce_logits(tf.ones_like(d_real), d_real)
                d_loss_fake = self.bce_logits(tf.zeros_like(d_fake), d_fake)
                d_loss = d_loss_real + d_loss_fake

            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
            d_grads = [tf.clip_by_norm(g, 0.3) for g in d_grads]
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        
        else: 
            z_real = tf.random.normal((batch_size, self.latent_dim))
            z_fake = tf.stop_gradient(self.encoder(data[0], training=False))
            z_real_noisy = z_real + tf.random.normal(tf.shape(z_real), stddev=0.02)
            z_fake_noisy = z_fake + tf.random.normal(tf.shape(z_fake), stddev=0.02)
            d_real = self.discriminator(z_real_noisy, training=False)
            d_fake = self.discriminator(z_fake_noisy, training=False)
            d_loss_real = self.bce_logits(tf.ones_like(d_real), d_real)
            d_loss_fake = self.bce_logits(tf.zeros_like(d_fake), d_fake)
            d_loss = d_loss_real + d_loss_fake

        total_ae_loss = total_recon_loss + total_g_loss + d_loss
        self.train_loss_tracker.update_state(total_ae_loss)
        self.train_recon_loss_tracker.update_state(reconstruction_loss)
        self.d_train_loss_tracker.update_state(d_loss)
        self.g_train_loss_tracker.update_state(g_loss)

        return {
            "loss": self.train_loss_tracker.result(),
            "recon_loss": self.train_recon_loss_tracker.result(),
            "d_loss": self.d_train_loss_tracker.result(),
            "g_loss": self.g_train_loss_tracker.result(),
            "adv_weight": adversarial_weight,
            "latent_mean_norm": float(z_mean_norm),
            "latent_mean_std": float(tf.reduce_mean(z_std)),
            "latent_coverage": float(z_coverage)
        }

    def test_step(self, data):
        batch_size = tf.shape(data[0])[0]
        adversarial_weight = self.get_adversarial_weight()
        z = self.encoder(data[0], training=False)
        z_mean = tf.reduce_mean(z, axis=0) 
        z_mean_norm = tf.norm(z_mean) 
        z_std = tf.math.reduce_std(z, axis=0)
        z_coverage = tf.reduce_mean(tf.cast(tf.abs(z) < 3.0, tf.float32)) 
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss = rmse(data[0], reconstruction)
        z_var = tf.math.reduce_variance(z, axis=0)
        mom_loss = 0.1 * (tf.reduce_mean(tf.square(z_mean)) +
                            tf.reduce_mean(tf.abs(z_var - 1.0)))
        mean_centering_loss = 0.1 * tf.reduce_mean(tf.abs(tf.reduce_mean(z, axis=0)))
        total_recon_loss = self.reconstruction_weight * reconstruction_loss + mom_loss + mean_centering_loss
        z_real = tf.random.normal(shape=(batch_size, self.latent_dim), mean=0.0, stddev=1.0)
        z_real_noisy = z_real + tf.random.normal(tf.shape(z_real), stddev=0.02)
        z_fake_noisy = z + tf.random.normal(shape=tf.shape(z), mean=0.0, stddev=0.02)
        d_real = self.discriminator(z_real_noisy, training=False)
        d_fake = self.discriminator(z_fake_noisy, training=False)
        d_loss_real = self.bce_logits(tf.ones_like(d_real), d_real)
        d_loss_fake = self.bce_logits(tf.zeros_like(d_fake), d_fake)
        d_loss = d_loss_real + d_loss_fake
 
        z_noisy = z + tf.random.normal(tf.shape(z), stddev=0.01)
        g_loss = self.bce_logits(tf.ones((batch_size, 1)), 
                        self.discriminator(z_noisy, training=False))
        total_g_loss = adversarial_weight * g_loss
        total_ae_loss = total_recon_loss + total_g_loss + d_loss
        self.val_loss_tracker.update_state(total_ae_loss)
        self.val_recon_loss_tracker.update_state(reconstruction_loss)
        self.val_d_train_loss_tracker.update_state(d_loss)
        self.val_g_train_loss_tracker.update_state(g_loss)

        return {
            "loss": self.val_loss_tracker.result(),
            "recon_loss": self.val_recon_loss_tracker.result(),
            "d_loss": self.val_d_train_loss_tracker.result(),
            "g_loss": self.val_g_train_loss_tracker.result(),
            "latent_mean_norm": float(z_mean_norm),
            "latent_mean_std": float(tf.reduce_mean(z_std)),
            "latent_coverage": float(z_coverage)
        }
class EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'current_epoch'):
            self.model.current_epoch.assign(epoch)
        if hasattr(self.model, 'reset_discriminator_if_needed'):
            self.model.reset_discriminator_if_needed(epoch)