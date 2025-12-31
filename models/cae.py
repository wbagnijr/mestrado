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
    latent = layers.Dense(latent_dim, name='latent_layer')(x)
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

class CAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.train_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def call(self, x):
      return self.decoder(self.encoder(x))

    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.val_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = rmse(data[0], reconstruction)
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.train_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
      z = self.encoder(data[0], training=False)
      reconstruction = self.decoder(z, training=False)
      val_loss = rmse(data[0], reconstruction)
      self.val_loss_tracker.update_state(val_loss)
      return {"loss": self.val_loss_tracker.result() }