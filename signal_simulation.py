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
    init_amp = 2.0
    init_freq = 5.0
    init_amp2 = 1.0
    init_freq2 = 10.0
    
    num_samples = int(window_seconds * fs)
    t_plot = np.linspace(0, window_seconds, num_samples)
    
    data_buffer = np.zeros(num_samples)

    fig, (ax_time, ax_freq_plot) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.35, hspace=0.4)
    
    
    line_time, = ax_time.plot(t_plot, data_buffer, lw=2)
    ax_time.set_xlim(0, window_seconds)
    ax_time.set_ylim(-10, 10)
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

    ax_amp = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_freq = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_amp2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor='lightblue')
    ax_freq2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightblue')

    samp = Slider(ax_amp, 'Amplitud 1', 0.0, 10.0, valinit=init_amp)
    sfreq = Slider(ax_freq, 'Frecuencia 1 (Hz)', 0.1, 20.0, valinit=init_freq)
    samp2 = Slider(ax_amp2, 'Amplitud 2', 0.0, 10.0, valinit=init_amp2)
    sfreq2 = Slider(ax_freq2, 'Frecuencia 2 (Hz)', 0.1, 20.0, valinit=init_freq2)

    sim_state = {
        'current_time': 0.0,
        'dt': 1.0 / fs
    }

    def update(frame):
        amp = samp.val
        freq = sfreq.val
        amp2 = samp2.val
        freq2 = sfreq2.val
        
        val = (amp * np.sin(2 * np.pi * freq * sim_state['current_time']) +
               amp2 * np.sin(2 * np.pi * freq2 * sim_state['current_time']))
        
        data_buffer[:-1] = data_buffer[1:]
        data_buffer[-1] = val
        
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
