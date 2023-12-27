# show a list of images
from matplotlib import pyplot as plt


def show_images(images, ncols=8):
    nrows = len(images) // ncols
    if len(images) % ncols != 0:
        nrows += 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for i, image in enumerate(images):
        ax = axes[i // ncols, i % ncols]
        ax.imshow(image,)
        ax.axis('off')
    plt.show()