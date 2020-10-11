import numpy as np
import heapq as heap


CLUSTER_NO = 0


def getNeighbors(eps, obj_point, points_object):
    eps_neighborhood = []
    for point in points_object:
        dist = obj_point.distance(point)
        if dist <= eps:
            eps_neighborhood.append(point)
    return eps_neighborhood


def update_ordered_seed(neighbors, point, ordered_seed):
    core_dist = point.core_dist
    for neighbor in neighbors:
        if not neighbor.visited:
            new_reach_dist = max(core_dist, point.distance(neighbor))
            if np.math.isinf(neighbor.reachability_dist):
                neighbor.reachability_dist = new_reach_dist
                heap.heappush(ordered_seed, neighbor)
            else:
                if neighbor.reachability_dist > new_reach_dist:
                    neighbor.reachability_dist = new_reach_dist
                    heap.heapify(ordered_seed)


def expandClusterOrder(points_object, eps, minPts, ordered_objects, point):
    ordered_seed = []
    neighbors = getNeighbors(eps, point, points_object)
    point.visited = True
    point.setCoreDistance(neighbors, eps, minPts)
    ordered_objects.append(point)
    if not np.math.isinf(point.core_dist):
        update_ordered_seed(neighbors, point, ordered_seed)
        while len(ordered_seed) > 0:
            next_point = heap.heappop(ordered_seed)
            neighbors = getNeighbors(eps, next_point, points_object)
            next_point.visited = True
            next_point.setCoreDistance(neighbors, eps, minPts)
            ordered_objects.append(next_point)
            if not np.math.isinf(next_point.core_dist):
                update_ordered_seed(neighbors, next_point, ordered_seed)


def extractDbscanClustering(ordered_objects, eps_tag):
    global CLUSTER_NO
    for point_obj in ordered_objects:
        if point_obj.reachability_dist > eps_tag:
            if point_obj.core_dist <= eps_tag:
                CLUSTER_NO = CLUSTER_NO + 1
                point_obj.clusterID = CLUSTER_NO
        else:
            point_obj.clusterID = CLUSTER_NO
