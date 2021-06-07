import matplotlib.pyplot as plt


def plot_array(array):

    for i in range(array.shape[2]):
        plt.imshow(array[:, :, i])
        plt.show()
        plt.close()
