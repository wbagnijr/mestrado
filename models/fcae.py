def encoder_model(latent_dim):
    encoder_inputs = tf.keras.Input(shape=(32, 32, 3))
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(768*3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(512*3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(256*3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    latent_layer = layers.Dense(latent_dim, name="latent_layer")(x)
    x = layers.BatchNormalization()(latent_layer)
    return tf.keras.Model(encoder_inputs, x, name="encoder")

def decoder_model(latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(256*3)(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(512*3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(768*3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(1024*3, activation="sigmoid")(x)
    decoder_outputs = layers.Reshape((32, 32, 3))(x)
    return tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

class FCAE(tf.keras.Model):
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