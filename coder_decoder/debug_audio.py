import numpy as np
from scipy.io import wavfile
from coder_decoder import SignalCoderDecoder, save_encoded_file, read_encoded_file

def test_L_wav():
    filename = "L.wav"
    try:
        rate, data = wavfile.read(filename)
    except FileNotFoundError:
        print(f"{filename} not found.")
        return

    print(f"Original: min={data.min()}, max={data.max()}, dtype={data.dtype}")
    
    # Use a small number of levels to make issues obvious, or a "standard" one like 256 (8-bit)
    levels = 256 
    coder = SignalCoderDecoder(levels)
    
    # Process
    if len(data.shape) > 1:
        data_mono = data.mean(axis=1) # Average channels
    else:
        data_mono = data
        
    indices, min_val, max_val = coder.quantization(data_mono)
    print(f"Quantized indices: min={indices.min()}, max={indices.max()}")
    
    # Test text file writing
    binary_data = coder.encode_to_binary(indices)
    tmp_filename = "debug_encoded.txt"
    save_encoded_file(tmp_filename, binary_data, min_val, max_val, rate, levels, "audio")
    
    # Test reading back
    read_bin, params = read_encoded_file(tmp_filename)
    decoded_indices = coder.decode_from_binary(read_bin)
    
    if len(decoded_indices) != len(indices):
        print("ERROR: Length mismatch!")
        
    if not np.array_equal(indices, decoded_indices):
        print("ERROR: Indices mismatch!")
        diff_count = np.sum(indices != decoded_indices)
        print(f"Count of different indices: {diff_count}")
    else:
        print("SUCCESS: Indices match perfectly after I/O.")
    
    reconstructed = coder.reconstruct(decoded_indices, params['MIN'], params['MAX'])
    print(f"Reconstructed: min={reconstructed.min()}, max={reconstructed.max()}")

    # Check error
    # We expect some error, but check for catastrophic failure
    error = np.abs(data_mono - reconstructed)
    print(f"Max absolute error: {error.max()}")
    print(f"Mean absolute error: {error.mean()}")

    # Compare first few samples
    print("First 10 samples comparison:")
    for i in range(10):
        print(f"Orig: {data_mono[i]:.4f}, Rec: {reconstructed[i]:.4f}, Idx: {indices[i]}")

if __name__ == "__main__":
    test_L_wav()
