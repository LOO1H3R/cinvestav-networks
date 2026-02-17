import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

def main():
    """Real-time signal simulation with interactive frequency and amplitude controls, displaying time and frequency domain representations."""
    fs = 100
    window_seconds = 2.0
    num_signals = 3
    if len(sys.argv) > 1:
        try:
            num_signals = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument '{sys.argv[1]}'. Using default: 3")
    else:
        try:
            choice = input("Enter number of signals (default 3): ")
            if choice.strip():
                num_signals = int(choice)
        except ValueError:
            print("Invalid input. Using default: 3")
    

    init_amps = [2.0 if i == 0 else 1.0 for i in range(num_signals)]
    init_freqs = [5.0 * (i + 1) for i in range(num_signals)]
    

    colors = ['lightgoldenrodyellow', 'lightblue', 'lightgreen', 'lightpink', 'thistle']

    num_samples = int(window_seconds * fs)
    t_plot = np.linspace(0, window_seconds, num_samples)
    
    data_buffer = np.zeros(num_samples)

    control_height_per_signal = 0.1
    bottom_margin = 0.05 + (num_signals * control_height_per_signal)
    
    fig, (ax_time, ax_freq_plot) = plt.subplots(2, 1, figsize=(10, 8 + (num_signals - 2)*0.5))
    plt.subplots_adjust(bottom=bottom_margin, hspace=0.4)
    
    
    line_time, = ax_time.plot(t_plot, data_buffer, lw=2)
    ax_time.set_xlim(0, window_seconds)

    ax_time.set_ylim(-5 * num_signals, 5 * num_signals)
    ax_time.set_title('Dominio del Tiempo')
    ax_time.set_ylabel('Amplitud')
    ax_time.grid(True)

    freqs = np.fft.fftfreq(num_samples, 1/fs)
    half_n = num_samples // 2
    freqs_pos = freqs[:half_n]
    
    line_freq, = ax_freq_plot.plot(freqs_pos, np.zeros(half_n), lw=2, color='r')
    ax_freq_plot.set_xlim(0, fs/2)
    ax_freq_plot.set_ylim(0, 10)
    ax_freq_plot.set_title('Dominio de la Frecuencia (FFT)')
    ax_freq_plot.set_xlabel('Frecuencia (Hz)')
    ax_freq_plot.set_ylabel('Magnitud')
    ax_freq_plot.grid(True)

    sliders = []


    for i in range(num_signals):
        base_y = 0.05 + (num_signals - 1 - i) * 0.10
        color = colors[i % len(colors)]

        ax_freq = plt.axes([0.25, base_y, 0.65, 0.03], facecolor=color)
        sfreq = Slider(ax_freq, f'Frecuencia {i+1} (Hz)', 0.1, 50.0, valinit=init_freqs[i])
        
        ax_amp = plt.axes([0.25, base_y + 0.05, 0.65, 0.03], facecolor=color)
        samp = Slider(ax_amp, f'Amplitud {i+1}', 0.0, 10.0, valinit=init_amps[i])
        
        sliders.append({'amp': samp, 'freq': sfreq})

    sim_state = {
        'current_time': 0.0,
        'dt': 1.0 / fs
    }

    def update(frame):
        total_val = 0.0
        
        for s in sliders:
            amp = s['amp'].val
            freq = s['freq'].val
            total_val += amp * np.sin(2 * np.pi * freq * sim_state['current_time'])
        
        data_buffer[:-1] = data_buffer[1:]
        data_buffer[-1] = total_val
        
        fft_vals = np.fft.fft(data_buffer)
        fft_mag = (2.0 / num_samples) * np.abs(fft_vals[:half_n])
        
        line_time.set_ydata(data_buffer)
        line_freq.set_ydata(fft_mag)
        
        sim_state['current_time'] += sim_state['dt']
        
        return line_time, line_freq

    
    ani = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
    
    plt.show()

if __name__ == "__main__":
    main()
