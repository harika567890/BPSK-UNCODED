import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_bits = 1000  # Number of bits to transmit
bit_rate = 1000  # Bit rate (bits per second)
frequency = 10   # Carrier frequency (Hz)
snr_db = 10      # Signal-to-Noise Ratio (dB)

# Generate random bits (0's and 1's)
bits = np.random.randint(2, size=num_bits)

# Modulate bits to BPSK symbols (0 -> -1, 1 -> 1)
symbols = 2 * bits - 1

# Time vector
t = np.arange(0, num_bits / bit_rate, 1 / bit_rate)

# Carrier signal
carrier = np.sin(2 * np.pi * frequency * t)

# Modulated signal
modulated_signal = symbols * carrier

# Add Gaussian noise
noise_power = 0.5 / (10 ** (snr_db / 10))  # calculate noise power
noise = np.sqrt(noise_power) * np.random.randn(len(modulated_signal))
received_signal = modulated_signal + noise

# Demodulation (coherent detection)
demodulated_signal = received_signal * carrier

# Integrate over one bit period and make decisions
bits_received = (np.sum(demodulated_signal.reshape(-1, len(carrier)), axis=1) > 0).astype(int)

# Calculate Bit Error Rate (BER)
ber = np.sum(bits != bits_received) / num_bits

print("Bit Error Rate (BER):", ber)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t[:100], carrier[:100])
plt.title('Carrier Signal')

plt.subplot(3, 1, 2)
plt.plot(t[:100], modulated_signal[:100])
plt.title('Modulated Signal')

plt.subplot(3, 1, 3)
plt.plot(range(len(bits_received)), bits_received, 'b.')
plt.title('Received Bits')

plt.tight_layout()
plt.show()
