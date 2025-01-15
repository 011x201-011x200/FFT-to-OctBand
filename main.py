import argparse
import fft
import plot_thirds

def validate_list(input_data):
    if not isinstance(input_data, list):
        raise ValueError(f"Expected a list, but got {type(input_data).__name__}")
    return input_data

def main():
    parser = argparse.ArgumentParser(description="Low Frequency Analysis of a wav file")
    parser.add_argument("--file", type=str, required=True, help="Path to .wav file")
    parser.add_argument("--to_thirds", type=bool, required=False, help="[optional] Calculate Average Thirds")
    parser.add_argument("--plot", type=int, required=False, help="[optional] Options: \n 1 -> Thirds/FFT Slider for whole file /n 2 -> Averaged Third Analysis")
    args = parser.parse_args()
    #2[51.2,50.3,50.6,53.6,52.2,50.1,49.0,49.1,46.3,41.4,41.1,43.1]
    #1[51.4, 52.4, 53.5, 54.0, 48.5, 48.4, 50.8, 50.3, 43.5, 45.7, 46.1, 44.0, 45.4, 42.4, 44.2, 44.0, 38.6]
    # subset data
    if not args.file:
        print("Error: Need filepath")
    else:
        analyzer = fft.WavFFTAnalyzer(args.file)
        frequencies = analyzer.fft_frequencies
        db_values = analyzer.fft_magnitudes
        if args.plot == 1:
            analyzer.plot_fft_and_octave_bands()
        
        if not args.to_thirds and args.plot == 2:
            print("Error: Need to_thirds for Averaged Thirds Analysis")
                
        
if __name__ == "__main__":
    main()
