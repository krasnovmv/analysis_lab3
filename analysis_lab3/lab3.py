import pandas
from sklearn import metrics
from sklearn.cluster import AffinityPropagation, KMeans


def matplotlib_show(df, ap, cluster_count):
    labels = ap.labels_

    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.close("all")
    plt.figure(1)
    plt.clf()

    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(cluster_count), colors):
        class_members = labels == k
        # cluster_center = df.values[cluster_centers_indices[k]]
        plt.plot(df.values[class_members, 0], df.values[class_members, 1], col + ".")
        # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=12)
        # for x in df.values[class_members]:
        #     plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title(f"Расчетное количество кластеров: {cluster_count}")
    plt.show()


def plotly_show(df, ap, cluster_count):
    labels = ap.labels_

    import plotly.graph_objects as go
    from itertools import compress

    fig = go.Figure()
    shapes = []

    for k in range(cluster_count):
        class_members = labels == k
        # cluster_center = df.values[cluster_centers_indices[k]]
        x = df.values[class_members, 0]
        y = df.values[class_members, 1]
        text = list(compress(df.index.values, class_members))
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name=f"Кластер {k + 1}", text=text)
        )

    layout = {
        "title": f"Расчетное количество кластеров: {cluster_count}",
        "xaxis": {"zeroline": False},
        "yaxis": {"zeroline": False},
    }
    fig.update_layout(layout, shapes=shapes)

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    fig.show()


def print_cluster_info(values, labels_true, labels):
    print("Гомогенность: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Полнота: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-мера: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print(
        "Скорректированный индекс схожести (ARI): %0.3f"
        % metrics.adjusted_rand_score(labels_true, labels)
    )
    print(
        "Скорректированная взаимная информация (AMI): %0.3f"
        % metrics.adjusted_mutual_info_score(
            labels_true, labels, average_method="arithmetic"
        )
    )
    print(
        "Силуэт выборки: %0.3f"
        % metrics.silhouette_score(values, labels, metric="sqeuclidean")
    )


def affinity_propagation(df):
    ap = AffinityPropagation(damping=0.90).fit(df.values)
    labels_true = df.index.values
    labels = ap.labels_

    print('МЕТОД РАСПРОСТРАНЕНИЯ БЛИЗОСТИ: ')
    print("Расчетное количество кластеров: %d" % len(ap.cluster_centers_indices_))
    print_cluster_info(df.values, labels_true, labels)

    matplotlib_show(df, ap, len(ap.cluster_centers_indices_))
    plotly_show(df, ap, len(ap.cluster_centers_indices_))


def k_means(df):
    cluster_count = 11
    km = KMeans(n_clusters=cluster_count).fit(df.values)
    labels_true = df.index.values
    labels = km.labels_

    print('МЕТОД K БЛИЖАЙШИХ СОСЕДЕЙ: ')
    print_cluster_info(df.values, labels_true, labels)

    matplotlib_show(df, km, cluster_count)
    plotly_show(df, km, cluster_count)


def main():
    file = pandas.read_csv("cropped_data.csv", index_col=0, na_values="нет данных")
    df = file.dropna(0)

    affinity_propagation(df)
    k_means(df)


if __name__ == "__main__":
    main()
