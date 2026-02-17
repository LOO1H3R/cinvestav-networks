import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os

# Ensure we can use plots (fallback to Agg if no display found, but try TkAgg if specified)
try:
    import matplotlib
    # matplotlib.use('TkAgg') # Uncomment if running locally with Tkinter support
except:
    pass

class SignalCoderDecoder:
    def __init__(self, levels):
        self.levels = levels
        self.bits = int(np.ceil(np.log2(levels)))
    
    def quantization(self, signal, min_val=None, max_val=None):
        """
        Quantize signal into N discrete levels.
        Maps min_val -> 0 and max_val -> levels-1.
        """
        signal = np.array(signal)
        
        # Determine dynamic range if not provided
        if min_val is None:
            min_val = np.min(signal)
        if max_val is None:
            max_val = np.max(signal)
            
        # Avoid division by zero if signal is flat
        if max_val == min_val:
            return np.zeros_like(signal, dtype=int), min_val, max_val
            
        # Normalize signal to [0, 1]
        signal_norm = (signal - min_val) / (max_val - min_val)
        
        # Clip just in case
        signal_norm = np.clip(signal_norm, 0, 1)
        
        # Map to 0..levels-1 evenly
        # Uses round to utilize full range of bins more symmetrically
        # 0.0 -> 0, 1.0 -> levels-1
        quant_indices = np.round(signal_norm * (self.levels - 1))
        quant_indices = np.clip(quant_indices, 0, self.levels - 1).astype(int)
        
        return quant_indices, min_val, max_val

    def reconstruct(self, quant_indices, min_val, max_val):
        """
        Reconstruct signal from quantized indices.
        Maps 0 -> min_val and levels-1 -> max_val.
        """
        if self.levels <= 1:
            return np.full_like(quant_indices, min_val, dtype=float)
            
        # Linear interpolation from index 0..(L-1) to min..max
        step = (max_val - min_val) / (self.levels - 1)
        reconstructed = min_val + (quant_indices * step)
        return reconstructed

    def encode_to_binary(self, quant_indices):
        """Convert indices to binary strings."""
        binary_format = f'{{0:0{self.bits}b}}'
        # Ensure input is iterable
        if np.ndim(quant_indices) == 0:
            quant_indices = [quant_indices]
        return [binary_format.format(x) for x in quant_indices]
    
    def decode_from_binary(self, binary_list):
        """Convert binary strings back to indices."""
        return np.array([int(b, 2) for b in binary_list])

def save_encoded_file(filename, binary_data, min_val, max_val, sample_rate, levels, channels=1, signal_type="analog"):
    """Save binary data and metadata to a text file."""
    print(f"Saving encoded data to {filename}...")
    with open(filename, 'w') as f:
        # Header with metadata for reconstruction
        f.write(f"# TYPE={signal_type}\n")
        f.write(f"# MIN={min_val}\n")
        f.write(f"# MAX={max_val}\n")
        f.write(f"# LEVELS={levels}\n")
        f.write(f"# RATE={sample_rate}\n")
        f.write(f"# CHANNELS={channels}\n")
        f.write(f"# SAMPLES={len(binary_data)}\n")
        f.write("# DATA_START\n")
        for b in binary_data:
            f.write(f"{b}\n")
    print("Save complete.")

def read_encoded_file(filename):
    """Read binary data and metadata from text file."""
    print(f"Reading encoded data from {filename}...")
    params = {}
    binary_data = []
    reading_data = False
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("# DATA_START"):
                reading_data = True
                continue
            
            if not reading_data and line.startswith("#"):
                # Parse header
                try:
                    key, val = line[1:].split("=")
                    key = key.strip()
                    if key in ["LEVELS", "RATE", "SAMPLES", "CHANNELS"]:
                        params[key] = int(val)
                    elif key in ["MIN", "MAX"]:
                        params[key] = float(val)
                    else:
                        params[key] = val
                except:
                    pass
            elif reading_data:
                binary_data.append(line)
    
    # Defaults
    if 'CHANNELS' not in params:
        params['CHANNELS'] = 1
        
    return binary_data, params

def generate_sine_wave(freq=5, duration=1.0, rate=1000):
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    # create a sine wave signal with amplitude 1.0 (range -1 to 1)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal, rate

def plot_comparison(original, reconstructed, t, title="Signal Comparison"):
    plt.figure(figsize=(10, 6))
    # If stereo, plot first channel
    if original.ndim > 1:
        plt.plot(t, original[:, 0], label='Original (Left/Mono)', alpha=0.7)
    else:
        plt.plot(t, original, label='Original', alpha=0.7)
        
    if reconstructed.ndim > 1:
        plt.plot(t, reconstructed[:, 0], label='Reconstructed (Left/Mono)', alpha=0.7, linestyle='--')
    else:
        plt.plot(t, reconstructed, label='Reconstructed', alpha=0.7, linestyle='--')
        
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- CONSTANTS for folders ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def run_demo_simple_signal(levels=16):
    print("\n--- DEMO: Simple Sine Wave ---")
    
    # 1. Generate Signal
    t, signal, rate = generate_sine_wave(freq=2, duration=1.0, rate=100)
    print(f"Generated Sine Wave: 2Hz, 1.0s, {rate}Hz sampling.")
    
    # 2. Encode
    coder = SignalCoderDecoder(levels)
    print(f"Encoding with {levels} levels ({coder.bits} bits)...")
    indices, min_val, max_val = coder.quantization(signal)
    binary_data = coder.encode_to_binary(indices)
    
    # 3. Save to output folder
    filename = os.path.join(OUTPUT_DIR, "signal_encoded.txt")
    save_encoded_file(filename, binary_data, min_val, max_val, rate, levels, channels=1, signal_type="simple")
    
    # 4. Decode
    read_bin, params = read_encoded_file(filename)
    decoded_indices = coder.decode_from_binary(read_bin)
    reconstructed = coder.reconstruct(decoded_indices, params['MIN'], params['MAX'])
    
    # 5. Plot
    print("Plotting results (close window to continue)...")
    plot_comparison(signal, reconstructed, t, f"Sine Wave (L={levels})")

def run_demo_audio(input_wav_path, levels=64):
    print(f"\n--- DEMO: Audio File ({input_wav_path}) ---")
    
    if not os.path.exists(input_wav_path):
        print(f"Error: File {input_wav_path} not found.")
        return

    # 1. Read Wav
    rate, data = wavfile.read(input_wav_path)
    print(f"Read audio: Rate={rate}, Shape={data.shape}, Dtype={data.dtype}")
    
    channels = 1
    if len(data.shape) > 1:
        channels = data.shape[1]
    
    original_data = data.astype(float)
    
    # Flatten if stereo to treat as one long stream
    original_flat = original_data.flatten()
    
    # 2. Encode
    coder = SignalCoderDecoder(levels)
    print(f"Encoding with {levels} levels ({coder.bits} bits)...")
    indices, min_val, max_val = coder.quantization(original_flat)
    binary_data = coder.encode_to_binary(indices)
    
    # 3. Save to Output Folder
    base_name = os.path.basename(input_wav_path)
    name_no_ext = os.path.splitext(base_name)[0]
    
    # Create subfolder for this file
    file_output_dir = os.path.join(OUTPUT_DIR, name_no_ext)
    if not os.path.exists(file_output_dir):
        os.makedirs(file_output_dir)
        print(f"Created output directory: {file_output_dir}")

    txt_filename = os.path.join(file_output_dir, f"{name_no_ext}_encoded.txt")
    
    save_encoded_file(txt_filename, binary_data, min_val, max_val, rate, levels, channels=channels, signal_type="audio")
    
    # 4. Decode
    read_bin, params = read_encoded_file(txt_filename)
    decoded_indices = coder.decode_from_binary(read_bin)
    
    rec_flat = coder.reconstruct(decoded_indices, params['MIN'], params['MAX'])
    
    # Reshape back to original channels
    if params['CHANNELS'] > 1:
        reconstructed = rec_flat.reshape((-1, params['CHANNELS']))
    else:
        reconstructed = rec_flat
        
    # 5. Save Output Wav
    out_wav_path = os.path.join(file_output_dir, f"{name_no_ext}_decoded.wav")
    print(f"Saving reconstructed audio to {out_wav_path}...")
    
    # Scale back to original integer type if data was integer
    if np.issubdtype(data.dtype, np.integer):
        rec_to_save = np.round(reconstructed).astype(data.dtype)
    else:
        rec_to_save = reconstructed.astype(data.dtype)
        
    wavfile.write(out_wav_path, int(params['RATE']), rec_to_save)
    
    # Plot segment (first 5 seconds)
    print("Plotting first 5.0 seconds of first channel...")
    target_seconds = 5.0
    num_samples = min(len(original_data), int(rate * target_seconds))
    t = np.arange(num_samples) / rate
    
    # Slice for plotting
    org_slice = original_data[:num_samples] if original_data.ndim == 1 else original_data[:num_samples, :]
    rec_slice = reconstructed[:num_samples] if reconstructed.ndim == 1 else reconstructed[:num_samples, :]
    
    plot_comparison(org_slice, rec_slice, t, f"Audio Segment: {base_name} (L={levels})")

def list_input_files():
    """Returns list of wav files in input dir."""
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return []
    
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.wav')]
    files.sort()
    return files

if __name__ == "__main__":
    # Ensure dirs exist
    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("=== Analog to Digital Encoder/Decoder Demo ===")
    print(f"Script Directory: {SCRIPT_DIR}")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    while True:
        print("\nOptions:")
        print("1. Encode/Decode Simple Sine Wave")
        print("2. Encode/Decode Audio (List Input Files)")
        print("3. Exit")
        
        choice = input("Select an option (1-3): ").strip()
        
        if choice == '1':
            try:
                l = int(input("Enter number of levels (e.g., 4, 8, 16, 256): ").strip())
                run_demo_simple_signal(levels=l)
            except ValueError:
                print("Invalid input for levels.")
                
        elif choice == '2':
            files = list_input_files()
            if not files:
                print(f"No .wav files found in '{INPUT_DIR}'.")
                print("Please add .wav files to that folder.")
            else:
                print("\nAvailable files:")
                for i, f in enumerate(files):
                    print(f"{i+1}. {f}")
                print(f"{len(files)+1}. Back to Menu")
                
                sel = input("Select file number: ").strip()
                try:
                    sel_idx = int(sel) - 1
                    if 0 <= sel_idx < len(files):
                        filename = files[sel_idx]
                        full_path = os.path.join(INPUT_DIR, filename)
                        
                        try:
                            l = int(input("Enter number of levels (e.g. 16, 64, 256): ").strip())
                            run_demo_audio(full_path, levels=l)
                        except ValueError:
                            print("Invalid input for levels.")
                    elif sel_idx == len(files):
                        pass # Back
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
                
        elif choice == '3':
            print("Exiting.")
            break
        else:
            print("Unknown option.")

