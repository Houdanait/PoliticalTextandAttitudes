import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_distribution(similarities, title, color, threshold):
    sorted_similarities = np.sort(similarities)[::1]
    cumulative_count = np.arange(1, len(sorted_similarities) + 1)
    
    plt.plot(sorted_similarities, cumulative_count, linestyle='-', color=color)
    plt.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold')
    
    # Calculate the number of dimensions >= threshold
    num_dimensions_above_threshold = np.sum(similarities >= threshold)
    
    plt.title(f"{title}\n{num_dimensions_above_threshold} dimensions w/cosine similarity > {threshold}")
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Dimensions >= Similarity')
    plt.grid(True)
    plt.legend()