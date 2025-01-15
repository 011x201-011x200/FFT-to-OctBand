import scipy.io.wavfile as wavfile
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import get_window
import plot_thirds


class WavFFTAnalyzer:
    def __init__(self, wavfile_path):
        # Read .wav file
        self.fs_rate, self.signal = wavfile.read(wavfile_path)
        print("Frequency sampling", self.fs_rate)
        l_audio = len(self.signal.shape)
        print("Channels", l_audio)

        # Stereo -> mono
        if len(self.signal.shape) == 2:
            self.signal = self.signal.sum(axis=1) / 2

        # Normalize
        if self.signal.dtype != np.float32 and self.signal.dtype != np.float64:
            self.signal = self.signal / np.max(np.abs(self.signal))

        # 1/3 octave center frequencies
        self.octave_bands = [2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5, 16, 20, 25,
                             31.5, 40, 50, 63, 80, 100, 125]
        self.freq_limit = 125
        self.window_size = int(self.fs_rate * 1)  # 1-second window
        self.secs = len(self.signal) / float(self.fs_rate)

        # Precompute the FFT for the entire signal
        self.precompute_fft()

        # Boundary markers
        self.start_boundary = None
        self.end_boundary = None
        self.data1_frequencies = [20, 25, 31.5, 40.0, 50, 63, 80, 100, 125]
        self.data1_values = [78.5, 68.5, 59.5, 51.5, 44, 37.5, 31.5, 26.5, 22]
        self.data2_frequencies = [20, 25, 31.5, 40.0, 50, 63, 80, 100, 125]
        self.data2_values = [64, 54, 44.5, 36.7, 30, 25, 20.8, 17, 13]
        self.hs_frequencies = [8, 10, 12.5, 16, 20, 25, 31.5, 40.0, 50, 63, 80, 100]
        self.hs_values = [103, 95, 87, 79, 71, 63, 55.5, 48, 40.5, 33.5, 28, 23.5]
        self.bar_widths = [fc * (2 ** (1 / 6) - 2 ** (-1 / 6)) for fc in self.octave_bands]

    def precompute_fft(self):
        """Precompute the FFT for the entire signal."""
        window_size = self.window_size
        overlap = 0.5  # 50% overlap
        step = int(window_size * (1 - overlap))

        # Prepare arrays to store FFT results
        fft_frequencies = np.fft.rfftfreq(window_size, d=1 / self.fs_rate)
        fft_magnitudes = []

        # Compute FFT for each segment
        for start_idx in range(0, len(self.signal) - window_size + 1, step):
            segment = self.signal[start_idx:start_idx + window_size]
            fft_window = get_window("hann", window_size)
            fft_magnitude = np.abs(np.fft.rfft(segment * fft_window))
            fft_magnitudes.append(fft_magnitude)

        self.fft_frequencies = fft_frequencies
        self.fft_magnitudes = np.array(fft_magnitudes)

    def fft_to_thirds(self, fft_frequencies, fft_magnitudes):
        """Calculate LZeq, LZmax, LZmin for each 1/3 octave band.""" 
        LZeq, LZmax, LZmin = [], [], []

        for fc in self.octave_bands:
            f_low = fc * 2 ** (-1 / 6)
            f_high = fc * 2 ** (1 / 6)

            # Filter FFT data within the band
            indices = (fft_frequencies >= f_low) & (fft_frequencies <= f_high)
            band_magnitudes = fft_magnitudes[indices]

            # Calculate LZeq, LZmax, LZmin
            if len(band_magnitudes) > 0:
                p_rms = np.sqrt(np.mean(band_magnitudes ** 2))
                LZeq.append(20 * np.log10(p_rms))
                LZmax.append(20 * np.log10(np.max(band_magnitudes)))
                LZmin.append(20 * np.log10(np.min(band_magnitudes)))
            else:
                LZeq.append(None)
                LZmax.append(None)
                LZmin.append(None)

        return LZeq, LZmax, LZmin

    def update_plot(self, ax_fft, ax_octave, slider_val, fig, plot=False, debug=1):
        """Update FFT and 1/3 octave plots based on slider value."""
        if self.start_boundary:
            start_idx = int(self.start_boundary * self.fs_rate / self.window_size)
            end_idx = int(self.slider.val * self.fs_rate / self.window_size)
            #start_idx = int(self.start_boundary * self.fs_rate)
            #end_idx = int(self.slider.val * self.fs_rate)
            fft_magnitudes = self.fft_magnitudes[start_idx:end_idx][0] # PROBLEM
            if debug:
                print(f"start_boundary: {self.start_boundary}, slider_val: {self.slider.val}")
                print(f"start_idx: {start_idx}, end_idx: {end_idx}")
                print(f"fft_magnitudes shape: {fft_magnitudes.shape}")
                print(f"Sliced fft_magnitudes: {fft_magnitudes}")
        else:
            segment_idx = int(slider_val * self.fs_rate / self.window_size)
            fft_magnitudes = self.fft_magnitudes[segment_idx]
            if debug:
                print(f"segment idxd: {segment_idx}")
                print(f"fft_magnitudes shape: {fft_magnitudes.shape}")
                print(f"Sliced fft_magnitudes: {fft_magnitudes}")
        
        freqs_side = self.fft_frequencies[self.fft_frequencies <= self.freq_limit]
        magnitudes_side = fft_magnitudes[:len(freqs_side)]
        if debug:
            print(f"freqs side: {freqs_side}")
            print(f"magnitudes side: {self.fft_magnitudes.shape}") 
        
        # Update FFT plot
        ax_fft.cla()
        ax_fft.bar(freqs_side, magnitudes_side, width=0.1)
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('Amplitude')
        ax_fft.set_title('FFT of Audio Signal (0-125 Hz)')
        ax_fft.set_ylim(0, np.max(magnitudes_side) * 1.2)

        # Calculate and update 1/3 octave band energies
        LZeq, _, _ = self.fft_to_thirds(freqs_side, magnitudes_side)

        # Replace None with 0 for plotting
        LZeq_cleaned = [val if val is not None else 0 for val in LZeq]

        ax_octave.cla()
        print(self.bar_widths)

        ax_octave.bar(self.octave_bands, LZeq_cleaned, width=self.bar_widths)
        ax_octave.set_xlabel('1/3 Octave Band Mid Frequency (Hz)')
        ax_octave.set_ylabel('Energy (dB)')
        ax_octave.set_title('1/3 Octave Band Energy')
        ax_octave.set_xscale('log')
        ax_octave.set_xticks(self.octave_bands)
        ax_octave.set_xticklabels([f'{freq:.0f}' if freq % 1 == 0 else f'{freq:.1f}' for freq in self.octave_bands])

        fig.canvas.draw_idle()

        if plot:
            plot_thirds.plot_bar(self.octave_bands, LZeq_cleaned, self.data1_frequencies, self.data1_values, self.data2_frequencies, self.data2_values, self.hs_frequencies, self.hs_values)

    def set_start(self, event):
        """Set the start boundary for analysis."""
        self.start_boundary = self.slider.val
        print(f"Start boundary set at: {self.start_boundary}s")

    def reset_slider(self, event):
        """Reset the slider and plots to the beginning (0 seconds)."""
        self.slider.reset()
        self.update_plot(self.ax_fft, self.ax_octave, 0, self.fig)

    def plot_fft_and_octave_bands(self):
        # Initial plot setup
        self.fig, (self.ax_fft, self.ax_octave) = plt.subplots(2, 1, figsize=(8, 6))
        plt.subplots_adjust(left=0.1, bottom=0.25)

        # Plot initial segment
        self.update_plot(self.ax_fft, self.ax_octave, 0, self.fig)

        # Slider
        ax_slider = plt.axes([0.1, 0.1, 0.65, 0.05], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Time (s)', 0, self.secs - 1, valinit=0, valstep=0.1)

        def slider_update(val):
            self.update_plot(self.ax_fft, self.ax_octave, self.slider.val, self.fig)

        ax_set_start = plt.axes([0.8, 0.15, 0.1, 0.04])
        button_start = Button(ax_set_start, 'Set Start')
        button_start.on_clicked(self.set_start)

        ax_set_stop = plt.axes([0.8, 0.1, 0.1, 0.04])
        button_stop = Button(ax_set_stop, 'Set Stop')
        button_stop.on_clicked(lambda event: self.update_plot(self.ax_fft, self.ax_octave, round(self.slider.val), self.fig, plot=True))

        ax_reset = plt.axes([0.8, 0.05, 0.1, 0.04])
        button_reset = Button(ax_reset, 'Reset')
        button_reset.on_clicked(self.reset_slider)

        self.slider.on_changed(slider_update)

        plt.show()
