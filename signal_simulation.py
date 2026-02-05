import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

def main():
    # --- CONFIGURACIÓN INICIAL ---
    fs = 100            # Frecuencia de muestreo (Hz)
    window_seconds = 2.0 # Tamaño de la ventana de visualización (segundos)
    init_amp = 2.0      # Amplitud inicial componente 1
    init_freq = 5.0     # Frecuencia inicial componente 1 (Hz)
    init_amp2 = 1.0     # Amplitud inicial componente 2
    init_freq2 = 10.0   # Frecuencia inicial componente 2 (Hz)
    
    # Generar vector de tiempo para el eje X (estático para la ventana)
    num_samples = int(window_seconds * fs)
    t_plot = np.linspace(0, window_seconds, num_samples)
    
    # Buffer de datos para la señal (inicializado en 0)
    data_buffer = np.zeros(num_samples)

    # Configuración de la figura y los ejes (2 filas)
    fig, (ax_time, ax_freq_plot) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(bottom=0.35, hspace=0.4) # Espacio para sliders y entre graficas
    
    # --- Plot 1: Dominio del Tiempo ---
    line_time, = ax_time.plot(t_plot, data_buffer, lw=2)
    ax_time.set_xlim(0, window_seconds)
    ax_time.set_ylim(-10, 10)
    ax_time.set_title('Dominio del Tiempo')
    ax_time.set_ylabel('Amplitud')
    ax_time.grid(True)

    # --- Plot 2: Dominio de la Frecuencia (FFT) ---
    # Eje de frecuencias (solo mitad positiva)
    freqs = np.fft.fftfreq(num_samples, 1/fs)
    half_n = num_samples // 2
    freqs_pos = freqs[:half_n]
    
    line_freq, = ax_freq_plot.plot(freqs_pos, np.zeros(half_n), lw=2, color='r')
    ax_freq_plot.set_xlim(0, fs/2) # Nyquist Limit
    ax_freq_plot.set_ylim(0, 10)
    ax_freq_plot.set_title('Dominio de la Frecuencia (FFT)')
    ax_freq_plot.set_xlabel('Frecuencia (Hz)')
    ax_freq_plot.set_ylabel('Magnitud')
    ax_freq_plot.grid(True)

    # --- CONTROLES (SLIDERS) ---
    # Componente 1
    ax_amp = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_freq = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    # Componente 2
    ax_amp2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor='lightblue')
    ax_freq2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightblue')

    samp = Slider(ax_amp, 'Amplitud 1', 0.0, 10.0, valinit=init_amp)
    sfreq = Slider(ax_freq, 'Frecuencia 1 (Hz)', 0.1, 20.0, valinit=init_freq)
    samp2 = Slider(ax_amp2, 'Amplitud 2', 0.0, 10.0, valinit=init_amp2)
    sfreq2 = Slider(ax_freq2, 'Frecuencia 2 (Hz)', 0.1, 20.0, valinit=init_freq2)

    # Estado global de la simulación
    sim_state = {
        'current_time': 0.0,
        'dt': 1.0 / fs
    }

    def update(frame):
        # 1. Obtener valores actuales de los sliders
        amp = samp.val
        freq = sfreq.val
        amp2 = samp2.val
        freq2 = sfreq2.val
        
        # 2. Calcular el siguiente punto en el tiempo (señal compuesta)
        val = (amp * np.sin(2 * np.pi * freq * sim_state['current_time']) +
               amp2 * np.sin(2 * np.pi * freq2 * sim_state['current_time']))
        
        # 3. Actualizar el buffer de tiempo (rolling buffer)
        data_buffer[:-1] = data_buffer[1:]
        data_buffer[-1] = val
        
        # 4. Calcular FFT
        fft_vals = np.fft.fft(data_buffer)
        fft_mag = (2.0 / num_samples) * np.abs(fft_vals[:half_n])
        
        # 5. Actualizar gráficas
        line_time.set_ydata(data_buffer)
        line_freq.set_ydata(fft_mag)
        
        # 6. Incrementar tiempo simulación
        sim_state['current_time'] += sim_state['dt']
        
        return line_time, line_freq

    # Crear la animación
    # interval en ms. 1000/fs para intentar ir a tiempo real, o menos para ir rápido.
    ani = FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)
    
    plt.show()

if __name__ == "__main__":
    main()
