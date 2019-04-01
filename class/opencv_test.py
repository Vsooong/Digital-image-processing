import cv2
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
image=[[0,5,7,7,5,8,7,8],
       [7,2,6,2,6,5,6,8],
       [6,9,7,7,0,7,2,7],
       [6,6,1,7,6,7,7,5],
       [9,6,0,7,8,2,6,7],
       [2,8,8,2,7,6,7,8],
       [7,3,2,6,1,7,5,8],
       [9,9,5,6,7,7,7,7]]
image_arr=np.asarray(image).flatten()
def plot_histogram(data):
    labels=np.arange(data.min(),data.max()+1,1)
    hist = np.histogram(data)
    counts=hist[0]
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.grid(axis='y', alpha=0.75)
    for a, b in zip(labels, counts):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)
    plt.xlabel('Intensity')
    plt.show()


plot_histogram(image_arr)

