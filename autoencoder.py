"""
Goal = build autoencoder for audio file
    1. Get nn to train
    2. Show that low error (and visually inspect) for the STFTs
    3. Do iSTFT and audially compare 

Lower input dimension of audio
Use low # of channels, and pooling/stride in the network 

Build out simple network, train (local or collab) 

"""
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import torchaudio
import matplotlib.pyplot as plt


# INPUT HERE -----------------------------------------------------
train_file_list = ['train_file1.wav', 'train_file2.wav', 'train_file3.wav']
test_file_list = ['test_file1.wav', 'test_file2.wav', 'test_file3.wav']
output_file_list = ['output_file1.wav', 'output_file2.wav', 'output_file3.wav']


# CODE BELOW -----------------------------------------------------

# Define the audio autoencoder architecture
class AudioAutoencoder(nn.Module):
    def __init__(self, num_input_features, encoding_size):
        super(AudioAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (num_input_features // 4) * (num_input_features // 4), encoding_size),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 32 * (num_input_features // 4) * (num_input_features // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (32, num_input_features // 4, num_input_features // 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Set the audio settings
sample_rate = 44100
duration = 4
num_input_samples = sample_rate * duration
num_input_features = 1024  # Number of STFT features (bins)

# Create an instance of the audio autoencoder
encoding_size = 64
autoencoder = AudioAutoencoder(num_input_features, encoding_size)

# Create the STFT and iSTFT transforms
transform_stft = transforms.Spectrogram(n_fft=num_input_samples, hop_length=num_input_samples // num_input_features)
transform_istft = transforms.GriffinLim(n_fft=num_input_samples, hop_length=num_input_samples // num_input_features)


# Generate the input STFT batch from the training audio files
train_batch_size = len(train_file_list)
train_input_stft_batch = torch.stack([transform_stft(waveform)[0] for waveform, _ in [torchaudio.load(file) for file in train_file_list]])

# Perform forward pass through the autoencoder
train_output_stft_batch = autoencoder(train_input_stft_batch)

# Reconstruct audio from the output STFT batch
train_reconstructed_waveform_batch = torch.stack([transform_istft(output_stft) for output_stft in train_output_stft_batch])

# Calculate the reconstruction loss
train_reconstruction_loss = nn.MSELoss()
train_loss = train_reconstruction_loss(train_output_stft_batch, train_input_stft_batch)

# Save the training examples
for i in range(train_batch_size):
    torchaudio.save(output_file_list[i], train_reconstructed_waveform_batch[i], sample_rate)


# Generate the input STFT batch from the test audio files
test_batch_size = len(test_file_list)
test_input_stft_batch = torch.stack([transform_stft(waveform)[0] for waveform, _ in [torchaudio.load(file) for file in test_file_list]])

# Perform forward pass through the autoencoder
test_output_stft_batch = autoencoder(test_input_stft_batch)

# Reconstruct audio from the output STFT batch
test_reconstructed_waveform_batch = torch.stack([transform_istft(output_stft) for output_stft in test_output_stft_batch])

# Calculate the reconstruction loss
test_reconstruction_loss = nn.MSELoss()
test_loss = test_reconstruction_loss(test_output_stft_batch, test_input_stft_batch)

# Save the test examples
for i in range(test_batch_size):
    torchaudio.save(output_file_list[i], test_reconstructed_waveform_batch[i], sample_rate)


# Print the shapes and losses
print("Training examples:")
print("Input STFT shape:", train_input_stft_batch.shape)
print("Output STFT shape:", train_output_stft_batch.shape)
print("Reconstructed waveform shape:", train_reconstructed_waveform_batch.shape)
print("Reconstruction loss:", train_loss.item())

print("Test examples:")
print("Input STFT shape:", test_input_stft_batch.shape)
print("Output STFT shape:", test_output_stft_batch.shape)
print("Reconstructed waveform shape:", test_reconstructed_waveform_batch.shape)
print("Reconstruction loss:", test_loss.item())


# Plot the spectrograms of the original and reconstructed examples
fig, axs = plt.subplots(2 * train_batch_size, 2, figsize=(10, 6 * (2 * train_batch_size)))
for i in range(train_batch_size):
    original_train_spec = transforms.AmplitudeToDB()(train_input_stft_batch[i].unsqueeze(0)).squeeze(0)
    reconstructed_train_spec = transforms.AmplitudeToDB()(train_output_stft_batch[i].unsqueeze(0)).squeeze(0)

    axs[i, 0].imshow(original_train_spec, aspect='auto', origin='lower')
    axs[i, 0].set_title(f"Original Train Example {i+1}")
    axs[i, 0].set_xlabel("Time")
    axs[i,1].set_ylabel("Frequency")

    axs[i, 1].imshow(reconstructed_train_spec, aspect='auto', origin='lower')
    axs[i, 1].set_title(f"Reconstructed Train Example {i+1}")
    axs[i, 1].set_xlabel("Time")
    axs[i, 1].set_ylabel("Frequency")

for i in range(train_batch_size):
    original_test_spec = transforms.AmplitudeToDB()(test_input_stft_batch[i].unsqueeze(0)).squeeze(0)
    reconstructed_test_spec = transforms.AmplitudeToDB()(test_output_stft_batch[i].unsqueeze(0)).squeeze(0)

    axs[train_batch_size + i, 0].imshow(original_test_spec, aspect='auto', origin='lower')
    axs[train_batch_size + i, 0].set_title(f"Original Test Example {i+1}")
    axs[train_batch_size + i, 0].set_xlabel("Time")
    axs[train_batch_size + i, 0].set_ylabel("Frequency")

    axs[train_batch_size + i, 1].imshow(reconstructed_test_spec, aspect='auto', origin='lower')
    axs[train_batch_size + i, 1].set_title(f"Reconstructed Test Example {i+1}")
    axs[train_batch_size + i, 1].set_xlabel("Time")
    axs[train_batch_size + i, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
