import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from faker import Factory

from OPTICS import expandClusterOrder, extractDbscanClustering
from Point import Point


def unsupervised_validation(ordered_points):
    X = []
    labels = []
    for point in ordered_points:
        if point.clusterID != 0:
            X.append(point.features)
            labels.append(point.clusterID)
    silhouette_acc = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    print("the silhouette score is: {}".format(silhouette_acc))
    print("the davis-bouldin score is: {}".format(davies_bouldin))


def varInit(path, data, variables):
    index = -1
    for dataset in data:
        index += 1
        if dataset in path:
            ep, minPts, tag_eps = variables[index]
            return ep, minPts, tag_eps
    return -1, -1, -1


def plotting_bar_chart(ordered_objects):
    data = [pts_instance.reachability_dist if not np.math.isinf(pts_instance.reachability_dist) else eps for
            pts_instance in ordered_objects]
    X = [i + 1 for i in range(len(ordered_objects))]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 3])
    ax.bar(x=X, height=data, width=1)


def plotting_clusters(ordered_objects, color_gen):
    clusters_ids = [point_obj.clusterID for point_obj in ordered_objects]
    cluster_amount = max(clusters_ids)
    clusters = []
    for i in range(cluster_amount + 1):
        clusters.append([point_obj.features for point_obj in ordered_objects if point_obj.clusterID == i])
    clust_id = 0
    for clust in clusters:
        x = []
        y = []
        for pnt in clust:
            x.append(pnt[0])
            y.append(pnt[1])
        plt.plot(x, y, 'o', color=color_gen.hex_color() if clust_id != 0 else '#000000')
        clust_id = clust_id + 1


dataNames = ['moons', 'R15', 'aggregation']
variables = [[0.5, 10, 0.15], [2, 10, 0.45], [9, 10, 1.7]]

path = 'data/aggregation.csv'
dataset = pd.read_csv(path, delimiter=' ', engine='python')
dataset = np.array(dataset)

# normalize the data:
norm_dataset = StandardScaler().fit_transform(X=dataset)

ordered_objects = []
color_gen = Factory.create()
eps, minPts, tag_eps = varInit(path, dataNames, variables)

if (eps, minPts, tag_eps) == (-1, -1, -1):
    print("error: the dataset is not familiar to the system ")
    exit(-1)

# creating array with all Points objects from the dataset
Points_object = []
for instance in dataset:
    Points_object.append(Point(instance))

# the OPTIC main loop
for point in Points_object:
    if not point.visited:
        expandClusterOrder(Points_object, eps, minPts, ordered_objects, point)
extractDbscanClustering(ordered_objects, tag_eps)

unsupervised_validation(ordered_objects)
plotting_clusters(ordered_objects, color_gen)
plotting_bar_chart(ordered_objects)
plt.show()
