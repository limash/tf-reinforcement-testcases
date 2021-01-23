import matplotlib as mpl
import matplotlib.pyplot as plt
import ray


# @ray.remote(num_gpus=1)
def use_gpu():
    """
    Call to check ids of available GPUs:
    ray.init()
    use_gpu.remote()
    """
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))


def plot_2d_array(array, name):
    fig = plt.figure(1)
    # make a color map of fixed colors
    # cmap = mpl.colors.ListedColormap(['blue', 'black', 'red'])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(  # noqa
        'my_colormap',
        ['blue', 'black', 'red'],
        256
    )
    # bounds = [-6, -2, 2, 6]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)  # noqa

    # tell imshow about color map so that only set colors are used
    # img = plt.imshow(array, interpolation='nearest', cmap=cmap, norm=norm)
    img = plt.imshow(array, interpolation='nearest', cmap=cmap)

    # make a color bar
    # plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-5, 0, 5])
    plt.colorbar(img)
    # plt.show()
    fig.savefig("data/pictures/"+name+".png")
    plt.close(fig)
