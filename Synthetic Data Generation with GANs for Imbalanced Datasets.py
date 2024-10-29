import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from imblearn.metrics import classification_report_imbalanced

class CreditCardGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.scaler = StandardScaler()
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
    def build_generator(self, input_shape):
        model = models.Sequential([
            layers.Dense(256, input_dim=self.latent_dim),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            
            layers.Dense(1024),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(),
            
            layers.Dense(input_shape, activation='tanh')
        ])
        return model
    
    def build_discriminator(self, input_shape):
        model = models.Sequential([
            layers.Dense(512, input_dim=input_shape),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            layers.Dense(128),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    @tf.function
    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake samples
            generated_samples = self.generator(noise, training=True)
            
            # Get discriminator outputs
            real_output = self.discriminator(real_samples, training=True)
            fake_output = self.discriminator(generated_samples, training=True)
            
            # Calculate losses
            real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
            fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
            disc_loss = real_loss + fake_loss
            
            gen_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    def preprocess_data(self, data):
        """Preprocess the input data"""
        if 'Time' in data.columns:
            data = data.drop('Time', axis=1)
            
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, data, epochs=10000, batch_size=32, sample_interval=1000):
        print("Starting preprocessing...")
        # Preprocess data
        X_scaled, y = self.preprocess_data(data)
        
        print("Building models...")
        # Initialize models
        self.generator = self.build_generator(X_scaled.shape[1])
        self.discriminator = self.build_discriminator(X_scaled.shape[1])
        
        # Get fraud samples
        fraud_samples = X_scaled[y == 1]
        n_fraud = len(fraud_samples)
        print(f"Number of fraud samples for training: {n_fraud}")
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(fraud_samples)
        dataset = dataset.shuffle(buffer_size=n_fraud)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        
        print(f"Starting training for {epochs} epochs...")
        # Training loop
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch in dataset:
                g_loss, d_loss = self.train_step(batch)
                g_losses.append(float(g_loss))
                d_losses.append(float(d_loss))
                
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
                
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}/{epochs}, D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
    
    def generate_synthetic_samples(self, n_samples):
        """Generate synthetic fraud samples"""
        print(f"Generating {n_samples} synthetic samples...")
        
        batch_size = 1000
        synthetic_samples = []
        
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            noise = tf.random.normal([current_batch_size, self.latent_dim])
            batch_samples = self.generator(noise, training=False)
            synthetic_samples.append(batch_samples.numpy())
        
        synthetic_samples = np.vstack(synthetic_samples)
        
        synthetic_samples = self.scaler.inverse_transform(synthetic_samples)
        print("Synthetic sample generation complete!")
        return synthetic_samples
    
def load_and_prepare_data(file_path):
    """Load and prepare the credit card dataset"""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(f"Dataset shape: {data.shape}")
        print(f"Number of fraud cases: {sum(data['Class'] == 1)}")
        print(f"Number of normal cases: {sum(data['Class'] == 0)}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
def evaluate_data_distribution(original_data, synthetic_data):
    """Compare distributions of original and synthetic data"""
    plt.figure(figsize=(15, 5))
    
    for i in range(min(5, original_data.shape[1])):
        plt.subplot(1, 5, i+1)
        plt.hist(original_data[:, i], bins=50, alpha=0.5, label='Original')
        plt.hist(synthetic_data[:, i], bins=50, alpha=0.5, label='Synthetic')
        plt.title(f'Feature {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    print("Loading data...")
    data = load_and_prepare_data('creditcard_2023.csv')
    
    if data is None:
        return None
    
    print("Initializing GAN...")
    # Initialize and train GAN
    gan = CreditCardGAN(latent_dim=100)
    
    print("Training GAN...")
    gan.train(data, epochs=10000, batch_size=32, sample_interval=100)
    
    # Generate synthetic samples
    print("Generating synthetic samples...")
    n_synthetic = sum(data['Class'] == 0) - sum(data['Class'] == 1)
    synthetic_samples = gan.generate_synthetic_samples(n_synthetic)
    
    # Evaluate the synthetic data
    print("Evaluating synthetic data...")
    original_fraud = data[data['Class'] == 1].drop(['Class', 'Time'], axis=1).values
    evaluate_data_distribution(original_fraud, synthetic_samples)
    
    print("Process complete!")
    return synthetic_samples

if __name__ == "__main__":
    synthetic_samples = main()