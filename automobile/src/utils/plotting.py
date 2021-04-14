import numpy as np
import pandas as pd
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # needed for waffle Charts

from scipy import linalg
from collections import OrderedDict


cmap_light = mpl.colors.ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
cmap_blue = mpl.colors.LinearSegmentedColormap.from_list(
    "", ["navy", "c", "cornflowerblue"])


def color_iter(iterations):
    color_list = ['navy', 'c', 'cornflowerblue', 'gold', 'darkorange']
    return itertools.cycle([color_list[i] for i in range(iterations)])


def plot_dataset(x_cords, y_cords, title):
    plt.scatter(x_cords, y_cords, label='Data')

    plt.xlim(np.amin(x_cords) - np.ptp(x_cords*0.05),
             np.amax(x_cords) + np.ptp(x_cords*0.05))
    plt.ylim(np.amin(y_cords) - np.ptp(y_cords*0.05),
             np.amax(y_cords) + np.ptp(y_cords*0.05))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_dataset_pcatsne(x, y, title, labels):
    colors = ["red", "green"]
    for index, point in enumerate(x):
        plt.scatter(point[0], point[1], color=colors[y[index]],
                    label=labels[y[index]])

    plt.xlim(np.amin(x[:, 0]) - np.ptp(x[:, 0]*0.05),
             np.amax(x[:, 0]) + np.ptp(x[:, 0]*0.05))
    plt.ylim(np.amin(x[:, 1]) - np.ptp(x[:, 1]*0.05),
             np.amax(x[:, 1]) + np.ptp(x[:, 1]*0.05))

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def plot_gaussian_cluster(X, Y_, means, covariances, index, title):
    fig = plt.figure(figsize=(6, 10))
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter(covariances.shape[0]))):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=20, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(np.amin(X[:, 0]) - np.ptp(X[:, 0]*0.05),
             np.amax(X[:, 0]) + np.ptp(X[:, 0]*0.05))
    plt.ylim(np.amin(X[:, 1]) - np.ptp(X[:, 1]*0.05),
             np.amax(X[:, 1]) + np.ptp(X[:, 1]*0.05))
    # plt.xticks(())
    # plt.yticks(())
    plt.title(title)
    plt.show()


def plot_gaussian_boundaries(X, y_pred, gmm, title):
    fig, ax = plt.subplots()
    # Decision Border
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, gmm, xx, yy, cmap=cmap_light, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y_pred)

    plt.xlim(np.amin(X[:, 0]) - np.ptp(X[:, 0]*0.05),
             np.amax(X[:, 0]) + np.ptp(X[:, 0]*0.05))
    plt.ylim(np.amin(X[:, 1]) - np.ptp(X[:, 1]*0.05),
             np.amax(X[:, 1]) + np.ptp(X[:, 1]*0.05))
    plt.title(title)
    plt.show()


def plot_kmeans_boundaries(x, y, y_pred, centroids, kmeans, title):
    fig, ax = plt.subplots()
    # Decision Border
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, kmeans, xx, yy, cmap=cmap_light, alpha=0.8)

    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                color="red", s=250, marker="*")
    plt.title(title)
    plt.show()


def plot_knn_boundaries(x, y_pred, neigh, title, labels):
    colors = {0: "red", 1: "green"}
    fig, ax = plt.subplots()
    # Decision Border
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, neigh, xx, yy, cmap=cmap_light, alpha=0.8)
    for index, point in enumerate(x):
        plt.scatter(point[0], point[1], color=colors[y_pred[index]],
                    label=labels[y_pred[index]])
    # plt.scatter(x[:, 0], x[:, 1], c=y_pred, label=y_pred)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def create_waffle_chart(categories, values, height, width, value_sign=''):
    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height  # total number of tiles
    # print('Total number of tiles is', total_num_tiles)

    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles)
                          for proportion in category_proportions]

    # print out number of tiles per category
    # for i, tiles in enumerate(tiles_per_category):
    #     print(df_dsn.index.values[i] + ': ' + str(tiles))

    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1

            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index

    # instantiate a new figure object
    fig = plt.figure(figsize=(6, 2))

    # use matshow to display the waffle chart
    colormap = cmap_blue
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

    # add gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + \
                ' (' + \
                str(np.round((values[i]/total_values)
                             * 100, 2)) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'

        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
