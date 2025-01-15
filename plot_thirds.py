import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bar(frequencies, db_values, data1_frequencies, data1_values, data2_frequencies, data2_values, hs_frequencies, hs_values):
    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(frequencies))

    plt.bar(x_positions, db_values, color='blue', alpha=0.6, width=0.4)  # Adjust bar width
    
    # Ensure the x-positions for the data frequencies are within the valid range
    data1_x_positions = [x_positions[frequencies.index(f)] for f in data1_frequencies if f in frequencies]
    data2_x_positions = [x_positions[frequencies.index(f)] for f in data2_frequencies if f in frequencies]
    hs_x_positions = [x_positions[frequencies.index(f)] for f in hs_frequencies if f in frequencies]

    # Plot the line graphs
    plt.plot(data1_x_positions, data1_values, color='purple', marker='o', label='HS 50% - ISO 226 (2006)', linestyle='-', markersize=8)
    plt.plot(data2_x_positions, data2_values, color='blue', marker='o', label='P1%-HS - ISO 28961 (2012)', linestyle='-', markersize=8)
    plt.plot(hs_x_positions, hs_values, color='red', marker='o', label='HS DIN 45 680', linestyle='-', markersize=8)
    
    plt.xticks(x_positions, frequencies, rotation=45)
    plt.xlabel('Terzbandmittenfrequenz (Hz)')
    plt.ylabel('LZeq (db(Z))')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""import pandas as pd
import matplotlib.pyplot as plt

# Function to read the CSV and plot bar charts for each row
def plot_bar_charts(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    data1 = pd.read_csv('test1.csv')
    data2 = pd.read_csv('test2.csv')
    hs = pd.read_csv('hs.csv')

    # Iterate over each row in the dataframe
    for index, row in data.iterrows():
        plt.figure(figsize=(10, 6))  # Create a new figure for each row
        print(data.columns[0:-2])
        print(row[0:-2])
        print(data.columns[-2:])
        print(row[-2:])
        plt.bar(data.columns[0:-2], row[0:-2])  # Bar plot (skipping the first column, which could be labels or IDs)
        plt.bar(data.columns[-2:], row[-2:], color='red')  # Bar plot (skipping the first column, which could be labels or IDs)
        plt.xlabel('Terzbandmittenfrequenz (Hz)')
        plt.ylabel('LZeq (db(Z))')
        for index, row in data1.iterrows():
            plt.plot(data1.columns[0:], row[0:], color='red', marker='o', label='P1%-HS ISO28961 (2012)', linestyle='--')
            #plt.text(7, 14, 'P1%-Hörschwelle ISO28961 (2012)', fontsize=8, color='black', ha='left')
        for index, row in data2.iterrows():
            plt.plot(data2.columns[0:], row[0:], color='purple', marker='o', label='HS 50% ISO226 (2006)', linestyle='--')
            #plt.text(7, 14, 'HS 50% ISO226 (2006)', fontsize=8, color='black', ha='left')
        for index, row in hs.iterrows():
            plt.plot(hs.columns[0:], row[0:], color='blue', marker='o', label='HS DIN 45 680', linestyle='--')
            #plt.text(7, 100, 'HS DIN 45 680', fontsize=8, color='black', ha='left')

        plt.legend()
    plt.show()  # Show the plot

# Function to add a line plot over the bar chart

# Example usage:
# Save this script and replace 'data.csv' with your actual CSV file
# Example line_values can be something like [10, 20, 30, 40, 50] corresponding to the number of columns
csv_file = 'test.csv'
plot_bar_charts(csv_file)  # To plot just bar charts
"""