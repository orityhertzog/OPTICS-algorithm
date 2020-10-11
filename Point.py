import numpy as np


class Point:
    def __init__(self, features_vector):
        self.core_dist = float('inf')
        self.reachability_dist = float('inf')
        self.visited = False
        self.features = features_vector
        self.clusterID = 0

    def distance(self, point2):
        return np.linalg.norm(self.features - point2.features)

    def setCoreDistance(self, neighbors, eps, minPts):
        if minPts <= len(neighbors):
            core_dist_points = []
            for neighbor in neighbors:
                core_dist_points.append(self.distance(neighbor))
            core_dist_points.sort()
            core_dist = core_dist_points[minPts - 1]
            if core_dist <= eps:
                self.core_dist = core_dist

    def __lt__(self, point2):
        return self.reachability_dist < point2.reachability_dist
