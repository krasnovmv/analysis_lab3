from itertools import compress

import numpy as np
import pandas
import plotly.graph_objects as go


def euclidean(u, v):
    diff = u - v
    return diff.dot(diff)


def manhattan_distance(u, v):
    return abs(u - v).sum()


def chebyshev_distance(u, v):
    return abs(u - v).max()


def percent_disagreement_distance(u, v):
    diff = u - v
    return np.count_nonzero(diff) / diff.size


def loss(data, distances_from_center, centers):
    loss_sum = 0
    for k in range(len(centers)):
        diff = data - centers[k]
        sq_distances = (diff * diff).sum(axis=1)
        loss_sum += (distances_from_center[:, k] * sq_distances).sum()
    return loss_sum


def plot_k_means(df, cluster_num, max_iter=20, beta=3.0, distance_func=euclidean, show_plots=False):
    data = df.values
    row_num, feature_num = data.shape
    distances_from_center = np.zeros((row_num, cluster_num))
    exponents = np.empty((row_num, cluster_num))

    initial_centers = np.random.choice(row_num, cluster_num, replace=False)
    centers = data[initial_centers]

    losses = []
    k = 0
    for i in range(max_iter):
        k += 1
        for k in range(cluster_num):
            for n in range(row_num):
                exponents[n, k] = np.exp(-beta * distance_func(centers[k], data[n]))
        distances_from_center = exponents / exponents.sum(axis=1, keepdims=True)

        centers = distances_from_center.T.dot(data) / distances_from_center.sum(axis=0, keepdims=True).T

        c = loss(data, distances_from_center, centers)
        losses.append(c)
        if i > 0:
            if np.abs(losses[-1] - losses[-2]) < 1e-5:
                break

    if show_plots:
        plotly_show(df, distances_from_center.argmax(axis=1), losses, cluster_num,
                    title=f"Кластеров {cluster_num} beta={beta} функция расстояния={distance_func.__name__}")

    print("Final loss", losses[-1])
    return centers, distances_from_center


def plotly_show(df, affiliation, losses, cluster_num, title=""):
    fig = go.Figure(data=[go.Scatter(y=losses, mode="lines")])
    fig.update_layout({
        "title": f"Функция потерь. {title}"})
    fig.show()

    fig = go.Figure()
    shapes = []

    for k in range(cluster_num):
        class_members = affiliation == k
        x = df.values[class_members, 0]
        y = df.values[class_members, 1]
        text = list(compress(df.index.values, class_members))
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name=f"Кластер {k + 1}", text=text)
        )

    layout = {
        "title": f"График. {title}",
        "xaxis": {"zeroline": False},
        "yaxis": {"zeroline": False},
    }
    fig.update_layout(layout, shapes=shapes)

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig.show()


def main():
    file = pandas.read_csv("cropped_data.csv", index_col=0, na_values="нет данных")
    df = file.dropna(0)

    show_plots = False
    plot_k_means(df, cluster_num=3, max_iter=20, beta=0.3, show_plots=show_plots)
    plot_k_means(df, cluster_num=4, max_iter=20, beta=0.35, show_plots=show_plots)
    plot_k_means(df, cluster_num=5, max_iter=20, beta=0.40, show_plots=show_plots)
    plot_k_means(df, cluster_num=6, max_iter=20, beta=0.45, show_plots=show_plots)
    plot_k_means(df, cluster_num=7, max_iter=20, beta=0.50, show_plots=show_plots)
    plot_k_means(df, cluster_num=8, max_iter=20, beta=0.3, show_plots=show_plots)
    plot_k_means(df, cluster_num=9, max_iter=20, beta=0.3, show_plots=show_plots)
    plot_k_means(df, cluster_num=11, max_iter=20, beta=0.3, show_plots=show_plots)
    plot_k_means(df, cluster_num=13, max_iter=20, beta=0.3, show_plots=show_plots)
    plot_k_means(df, cluster_num=15, max_iter=20, beta=0.3, show_plots=show_plots)


if __name__ == '__main__':
    main()
