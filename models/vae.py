class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def encoder_model(latent_dim):
    encoder_inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, strides=1, padding='same')(encoder_inputs)
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

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")  

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

    decoder_outputs = layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid')(x) 
    return tf.keras.Model(latent_inputs, decoder_outputs, name='decoder')

def vae_loss_total(data, reconstruction, z_log_var, z_mean, kl_weight=1.0):
    epsilon = 1e-7
    reconstruction_clipped = tf.clip_by_value(reconstruction, epsilon, 1.0 - epsilon)
    bce_per_element = - (data * tf.math.log(reconstruction_clipped) +
                        (1 - data) * tf.math.log(1 - reconstruction_clipped))
    reconstruction_loss = tf.reduce_sum(bce_per_element, axis=[1, 2, 3])
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(tf.clip_by_value(z_log_var, -10, 10)))
    kl_loss =  tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    weighted_kl_loss = kl_weight * kl_loss
    total_loss = 10 * reconstruction_loss + weighted_kl_loss 
    return total_loss, reconstruction_loss, kl_loss

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="val_kl_loss")

    def call(self, x):
        _, _, z = self.encoder(x)
        return self.decoder(z)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]
        
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            if isinstance(data, tuple):
               data = data[0]
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            total_loss, reconstruction_loss, kl_loss = vae_loss_total(data, reconstruction, z_log_var, z_mean, self.kl_weight)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        val_loss, val_reconstruction_loss, val_kl_loss = vae_loss_total(data, reconstruction, z_log_var, z_mean,  self.kl_weight)
        self.val_loss_tracker.update_state(val_loss)
        self.val_reconstruction_loss_tracker.update_state(val_reconstruction_loss)
        self.val_kl_loss_tracker.update_state(val_kl_loss)
        return {"loss": self.val_loss_tracker.result(),
                "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
                "kl_loss": self.val_kl_loss_tracker.result()}