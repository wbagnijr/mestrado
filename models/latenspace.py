class LatentSpaceMonitor(tf.keras.callbacks.Callback):
    def __init__(self, 
                 validation_data,
                 log_freq=5,
                 visualize=True,
                 num_samples=500,
                 save_dir=None,
                 pca_viz=True,
                 class_histograms=True,
                 show_reconstructions=False,
                 num_reconstruction_samples=4,
                 latent_extraction_fn=None):

        super().__init__()
        self.validation_data = validation_data
        self.log_freq = log_freq
        self.visualize = visualize
        self.num_samples = num_samples
        self.save_dir = save_dir or "./latent_viz"
        self.pca_viz = pca_viz
        self.class_histograms = class_histograms
        self.show_reconstructions = show_reconstructions
        self.num_reconstruction_samples = num_reconstruction_samples
        self.latent_extraction_fn = latent_extraction_fn
        
        self.val_batch = None
        self.val_labels = None
        self.detected_type = None
        self.last_metrics = {}
        
        if self.visualize and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._prepare_validation_batch()
    
    def _prepare_validation_batch(self):
        images_list = []
        labels_list = []
        collected_samples = 0

        for batch in self.validation_data.take(50):  
            if isinstance(batch, tuple):
                samples_needed = min(self.num_samples - collected_samples, batch[0].shape[0])
                if samples_needed <= 0:
                    break
                images_list.append(batch[0][:samples_needed])
                
                if len(batch) > 1: 
                    labels_list.append(batch[1][:samples_needed])
            else:
                samples_needed = min(self.num_samples - collected_samples, batch.shape[0])
                
                if samples_needed <= 0:
                    break
                    
                images_list.append(batch[:samples_needed])
            
            collected_samples += samples_needed
            
            if collected_samples >= self.num_samples:
                break
        
        if images_list:
            self.val_batch = tf.concat(images_list, axis=0)
            
            if labels_list:
                self.val_labels = tf.concat(labels_list, axis=0)
            else:
                self.val_labels = None
        else:
            self.val_batch = None
            self.val_labels = None
            print("Warning: No samples collected from validation dataset")
         
    def _extract_latent_vectors(self):
        z = self.model.encoder(self.val_batch)
        return {'latent': z}

    def _calculate_latent_metrics(self, latent_data):
        metrics = {}
        
        primary_latent = latent_data['latent'] 
        z_mean_per_dim = tf.reduce_mean(primary_latent, axis=0)
        z_mean_norm = tf.norm(z_mean_per_dim)
        z_std = tf.math.reduce_std(primary_latent, axis=0)
        z_avg_std = tf.reduce_mean(z_std)
        z_coverage = tf.reduce_mean(tf.cast(tf.abs(primary_latent) < 3.0, tf.float32))
        metrics['latent_z_mean'] = float(z_mean_norm)
        metrics['latent_z_std'] = float(z_avg_std)
        metrics['latent_z_coverage'] = float(z_coverage)

        latent_flat = tf.reshape(primary_latent, [tf.shape(primary_latent)[0], -1])
        latent_np = latent_flat.numpy()
        
   
        variance = tf.math.reduce_variance(latent_flat, axis=0)
        metrics['latent_variance_mean'] = float(tf.reduce_mean(variance))
        metrics['latent_variance_max'] = float(tf.reduce_max(variance))
        
        activation_mean = tf.reduce_mean(tf.abs(latent_flat))
        metrics['latent_activation_mean'] = float(activation_mean)
        
        sparsity = tf.reduce_mean(tf.cast(tf.abs(latent_flat) < 0.1, tf.float32))
        metrics['latent_sparsity'] = float(sparsity)
        
        significant_dims = tf.reduce_mean(tf.cast(variance > 0.01, tf.float32))
        metrics['latent_utilization'] = float(significant_dims)
        
        if len(latent_np) > 1:
            from sklearn.metrics.pairwise import euclidean_distances
            dists = euclidean_distances(latent_np)
            mask = ~np.eye(dists.shape[0], dtype=bool)
            
            metrics['latent_min_distance'] = float(np.min(dists[mask]))
            
            metrics['latent_mean_distance'] = float(np.mean(dists[mask]))
        
        return metrics, latent_flat
    
    def _visualize_latent_space(self, latent_data, metrics, latent_flat, epoch):
        if not self.visualize:
            return
            
        latent_np = latent_flat.numpy()
        
        plt.figure(figsize=(15, 12))
        plt.suptitle(f"Analise do Espaco Latente  Epoca {epoch+1}")
        
        plt.subplot(2, 3, 1)
        plt.hist(latent_np.flatten(), bins=50, alpha=0.7)
        plt.title("Distribuicao Global do Espaco Latente")
        plt.xlabel("Valor")
        plt.ylabel("Frequencia")
        
        plt.subplot(2, 3, 2)
        mean_by_dim = np.mean(latent_np, axis=0)
        plt.bar(range(len(mean_by_dim)), mean_by_dim, alpha=0.7)
        plt.title("Ativacao Media por Dimensao")
        plt.xlabel("Dimensao")
        plt.ylabel("Ativacao Media")
        
        plt.subplot(2, 3, 3)
        var_by_dim = np.var(latent_np, axis=0)
        plt.bar(range(len(var_by_dim)), var_by_dim, color='orange', alpha=0.7)
        plt.title("Variancia por Dimensao")
        plt.xlabel("Dimensao")
        plt.ylabel("Variancia")
        
        if self.val_labels is not None and self.class_histograms:
            plt.subplot(2, 3, 4)

            if len(self.val_labels.shape) > 1 and self.val_labels.shape[1] > 1:
                class_indices = np.argmax(self.val_labels.numpy(), axis=1)
            else:
                class_indices = self.val_labels.numpy().astype(int)
            
            top_dim = np.argmax(var_by_dim)
            
            unique_classes = np.unique(class_indices)
            for cls in unique_classes:
                values = latent_np[class_indices == cls, top_dim]
                mean_val = np.mean(values)
                std_val = np.std(values)
                n, bins, patches = plt.hist(values, bins=10, alpha=0.3, label=f'Class {cls}, Media: {mean_val:.3f}, Desvio: {std_val:.3f}')
                plt.axvline(mean_val, color=patches[0].get_facecolor(), linestyle='--')
       
            plt.title(f"Distribuicao por classe (Dim {top_dim})")
            plt.xlabel("Valor")
            plt.ylabel("Frequencia")
            plt.legend()

        if latent_np.shape[1] > 2 and self.pca_viz:
            plt.subplot(2, 3, 5)
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_np)
            
            if self.val_labels is not None:
                if len(self.val_labels.shape) > 1 and self.val_labels.shape[1] > 1:
                    colors = np.argmax(self.val_labels.numpy(), axis=1)
                else:
                    colors = self.val_labels.numpy()
                scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                             c=colors, cmap='tab10', alpha=0.8)
                plt.colorbar(scatter, label='Class')
            else:
                plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.8)

            plt.title(f"Visualizacao PCA (Var: {pca.explained_variance_ratio_.sum():.2f})")
            plt.xlabel("Componente 1")
            plt.ylabel("Componente 2")
        
        plt.subplot(2, 3, 6)
        plt.axis('off')
        metrics_text = "\n".join([
            f"Metricas do Espaco Latente:",
            f"Dimensao: {latent_np.shape[1]}",
            f"Media: {metrics['latent_z_mean']:.4f}",
            f"Desvio padrao: {metrics['latent_z_std']:.4f}",
            f"Cobertura: {metrics['latent_z_coverage']:.4f}",
            f"Variancia Media: {metrics['latent_variance_mean']:.4f}",
            f"Esparsidade: {metrics['latent_sparsity']:.4f}",
            f"Utilizacao: {metrics['latent_utilization']:.4f}",
        ])
        

        plt.text(0.1, 0.5, metrics_text, fontsize=10, va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'latent_analysis_epoch_{epoch+1}.png'))
        plt.close()

        if self.val_labels is not None and self.class_histograms:
            plt.figure(figsize=(15, 12))
            plt.suptitle(f"Analise do Espaco Latente - Epoca {epoch+1}", fontsize=16)
        
            if len(self.val_labels.shape) > 1 and self.val_labels.shape[1] > 1:
                class_indices = np.argmax(self.val_labels.numpy(), axis=1)
            else:
                class_indices = self.val_labels.numpy().astype(int)
            
            top_dim = np.argmax(var_by_dim)
            
            unique_classes = np.unique(class_indices)
            for cls in unique_classes:
                values = latent_np[class_indices == cls, top_dim]
                mean_val = np.mean(values)
                std_val = np.std(values)                
                n, bins, patches = plt.hist(values, bins=50, alpha=0.5, label=f'Class {cls}, Media: {mean_val:.3f}, Desvio: {std_val:.3f} ')
                x = np.linspace(min(values), max(values), 1000)
                pdf = norm.pdf(x, mean_val, std_val)
                bin_width = (max(values) - min(values)) / 50  
                scaling_factor = len(values) * bin_width  
                
                plt.plot(x, pdf * scaling_factor, color=patches[0].get_facecolor(), linewidth=3)
                plt.axvline(mean_val, color=patches[0].get_facecolor(), linestyle='--')
    
            
            plt.title(f"Distribuicao por classe (Dim {top_dim})")
            plt.xlabel("Valor")
            plt.ylabel("Frequencia")
            plt.legend()
    
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'latent_by_class_{epoch+1}.png'))
            plt.close()

        if self.show_reconstructions:
            plt.figure(figsize=(10, 3*min(4, self.num_reconstruction_samples)))
            plt.suptitle(f"Original vs. Reconstruction - Epoch {epoch+1}", fontsize=16)
            
            samples = self.val_batch[:self.num_reconstruction_samples]
            reconstructions = self.model(samples, training=False)
            
            for i in range(min(self.num_reconstruction_samples, len(samples))):
                plt.subplot(self.num_reconstruction_samples, 2, i*2+1)
                self._plot_image(samples[i].numpy(), "Original" if i == 0 else None)
                
                plt.subplot(self.num_reconstruction_samples, 2, i*2+2)
                self._plot_image(reconstructions[i].numpy(), "Reconstruction" if i == 0 else None)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'reconstructions_epoch_{epoch+1}.png'))
            plt.close()
    
    def _plot_image(self, img, title=None):
        if img.shape[-1] == 1:
            img = img.reshape(img.shape[:-1])
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
   
    def on_epoch_end(self, epoch, logs=None):
        latent_data = self._extract_latent_vectors()
        metrics, latent_flat = self._calculate_latent_metrics(latent_data)
        self.last_metrics = metrics  

        if (epoch + 1) % self.log_freq == 0:
            
            print(f"\nEpoca {epoch+1} - Analise do Espaco Latente:")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
            
            self._visualize_latent_space(latent_data, metrics, latent_flat, epoch)
        
        for name, value in self.last_metrics.items():
            logs[name] = value