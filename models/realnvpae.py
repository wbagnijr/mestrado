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

class AlphaAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 target_alpha=1.0, 
                 annealing_epochs=30,
                 warmup_epochs=20,  
                 start_alpha=0.0, 
                 verbose=True):
        super().__init__()
        self.target_alpha = target_alpha
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = annealing_epochs
        self.start_alpha = start_alpha
        self.verbose = verbose
        
        if self.ramp_epochs > 0:
            self.slope = (self.target_alpha - self.start_alpha) / self.ramp_epochs
        else:
            self.slope = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        current_alpha = 0.0

        if epoch < self.warmup_epochs:
            current_alpha = self.start_alpha
        elif epoch < (self.warmup_epochs + self.ramp_epochs):
            epoch_na_rampa = epoch - self.warmup_epochs
            current_alpha = self.start_alpha + self.slope * epoch_na_rampa
        else:
            current_alpha = self.target_alpha
            
        self.model.alpha.assign(current_alpha)
        logs['alpha'] = K.get_value(self.model.alpha)

class RealNVPKerasTemplate(tf.keras.Model):
    def __init__(self, hidden_units=(128, 128), **kwargs):
        super().__init__(**kwargs)
        self.hidden_layers = [
            tf.keras.layers.Dense(
                units, 
                activation="relu", 
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros"
            ) for units in hidden_units
        ]
        self.final_layer = None

    def call(self, x, output_units, **kwargs):
        x = tf.cast(x, tf.float32)
        for layer in self.hidden_layers:
            x = layer(x)
        if self.final_layer is None:
            self.final_layer = tf.keras.layers.Dense(
                output_units * 2,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="shift_and_log_scale"
            )

        y = self.final_layer(x)

        shift, log_scale = tf.split(y, 2, axis=-1)
        log_scale_safe = tf.tanh(log_scale) * 0.5
        return shift, log_scale_safe

def make_flow(latent_dim, num_coupling_layers=8):
    bijectors = []
    for i in range(num_coupling_layers):
        num_masked = latent_dim // 2
        
        template_model = RealNVPKerasTemplate(hidden_units=(256, 256))
        bijectors.append(
            tfb.RealNVP(
                num_masked=num_masked,
                shift_and_log_scale_fn=template_model
            )
        )
        bijectors.append(tfb.Permute(permutation=list(reversed(range(latent_dim)))))

    flow_bijector = tfb.Chain(list(reversed(bijectors)))
    base_distribution = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_dim), scale_diag=tf.ones(latent_dim))
    flow_dist = tfd.TransformedDistribution(distribution=base_distribution, bijector=flow_bijector)
    return flow_dist

class RealNVP_AE(tf.keras.Model):
    def __init__(self, encoder, decoder, flow_dist, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.flow_dist = flow_dist
        self.alpha = tf.Variable(alpha, trainable=False, dtype=tf.float32, name="alpha")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.neglogp_tracker = tf.keras.metrics.Mean(name="neg_log_p_z")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_recon_loss_tracker = tf.keras.metrics.Mean(name="val_recon_loss")
        self.val_neglogp_tracker = tf.keras.metrics.Mean(name="val_neg_log_p_z")
    
    def call(self, x):
      return self.decoder(self.encoder(x))

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.neglogp_tracker,self.val_total_loss_tracker, self.val_recon_loss_tracker, self.val_neglogp_tracker]

    @property
    def trainable_variables(self):
        ae_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        flow_vars = list(self.flow_dist.trainable_variables)
        return ae_vars + flow_vars

    @property
    def non_trainable_variables(self):
        ae_vars = self.encoder.non_trainable_variables + self.decoder.non_trainable_variables
        flow_vars = list(self.flow_dist.non_trainable_variables)
        return ae_vars + flow_vars  

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data[0], training=True)
            reconstruction = self.decoder(z, training=True)
            epsilon = 1e-7
            reconstruction_clipped = tf.clip_by_value(reconstruction, epsilon, 1.0 - epsilon)
            bce_per_element = - (data[0] * tf.math.log(reconstruction_clipped) +
                                (1 - data[0]) * tf.math.log(1 - reconstruction_clipped))
            
            reconstruction_loss = tf.reduce_sum(bce_per_element, axis=[1, 2, 3])
            recon_loss = tf.reduce_mean(reconstruction_loss)
            logp_z = self.flow_dist.log_prob(z)
            neglogp = -tf.reduce_mean(logp_z)
            loss = recon_loss + self.alpha * neglogp
        grads = tape.gradient(loss, self.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
       
        self.total_loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.neglogp_tracker.update_state(neglogp)       
    
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "neglogp": self.neglogp_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        z = self.encoder(data[0], training=False)
        reconstruction = self.decoder(z, training=False)
        epsilon = 1e-7
        reconstruction_clipped = tf.clip_by_value(reconstruction, epsilon, 1.0 - epsilon)
        bce_per_element = - (data[0] * tf.math.log(reconstruction_clipped) +
                            (1 - data[0]) * tf.math.log(1 - reconstruction_clipped))
        
        reconstruction_loss = tf.reduce_sum(bce_per_element, axis=[1, 2, 3])
        recon_loss = tf.reduce_mean(reconstruction_loss)
        logp_z = self.flow_dist.log_prob(z)
        neglogp = -tf.reduce_mean(logp_z)
        loss = recon_loss + self.alpha * neglogp

        self.val_total_loss_tracker.update_state(loss)
        self.val_recon_loss_tracker.update_state(recon_loss)
        self.val_neglogp_tracker.update_state(neglogp)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "recon_loss": self.val_recon_loss_tracker.result(),
            "neglogp": self.val_neglogp_tracker.result(),
        }