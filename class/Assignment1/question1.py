import cv2
import numpy as np
import matplotlib.pyplot as plt
image=[[0,5,7,7,5,8,7,8],
       [7,2,6,2,6,5,6,8],
       [6,9,7,7,0,7,2,7],
       [6,6,1,7,6,7,7,5],
       [9,6,0,7,8,2,6,7],
       [2,8,8,2,7,6,7,8],
       [7,3,2,6,1,7,5,8],
       [9,9,5,6,7,7,7,7]]
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
    plt.title('histogram')
    plt.show()

def getCDF(image_arr):
    hist=np.histogram(image_arr)[0]
    intensity=np.arange(image_arr.min(),image_arr.max()+1)
    cdf={}
    start=0
    assert hist.size==intensity.size
    for i in range(intensity.size):
        cdf[intensity[i]]=start+hist[i]
        start=cdf[intensity[i]]
    return cdf


def transformation(image_arr,max_Intensity):
    cdf=getCDF(image_arr)
    cdf_min= cdf[image_arr.min()]
    size=image_arr.size
    new_image=[]

    for i in image_arr:
        t_i=round((cdf[i]-cdf_min)/(size-cdf_min)*max_Intensity)
        new_image.append(t_i)
    return np.asarray(new_image)

def draw_matrix(image):
    import matplotlib.pylab as pl
    h,w=image.shape
    nx = w
    ny = h
    data = np.asarray(image,dtype=int)

    pl.figure(figsize=(5, 5))
    tb = pl.table(cellText=data, loc=(0, 0), cellLoc='center')

    tc = tb.properties()['child_artists']
    for cell in tc:
        cell.set_height(1 / ny)
        cell.set_width(1 / nx)

    ax = pl.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__=="__main__":
    image_arr = np.asarray(image).flatten()
    # plot_histogram(image_arr)

    new_image=transformation(image_arr, 9)
    # plot_histogram(new_image)
    new_image=new_image.reshape((8,8))
    draw_matrix(new_image)
