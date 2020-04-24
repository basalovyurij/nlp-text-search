from .dists import *
import heapq
import numpy as np
import random
from typing import Callable
from .vp_tree import _Node


class CustomVPTree:
    def __init__(self, dist: Dist):
        self.dist_fn = dist.dist
        self.dist_batch_fn = dist.batch_dist
        self.max_children = 2
        self.leaf_size = 5

    def set_dist(self, dist: Dist):
        self.dist_fn = dist.dist
        self.dist_batch_fn = dist.batch_dist

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.root, self.max_children, self.leaf_size

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.root, self.max_children, self.leaf_size = state

    def fit(self, points) -> None:
        items = [(item, ()) for item in points]
        # random.shuffle(items)
        self.root = self._make_node(items)

    def _make_node(self, items):
        if not items:
            return None

        node = _Node()

        node.lower_bounds = []
        node.upper_bounds = []
        for i in range(len(items[0][1])):
            distance_list = [item[1][i] for item in items]
            node.lower_bounds.append(min(distance_list))
            node.upper_bounds.append(max(distance_list))

        item = items.pop(self._get_middle_point_index(items))
        node.vantage = item[0]

        node.children = []

        if not items:
            return node

        #items = [(item[0], item[1] + (self.dist_fn(node.vantage, item[0]),)) for item in items]
        dists = self.dist_batch_fn([(node.vantage, item[0]) for item in items])
        items = [(item[0], item[1] + (dists[i],)) for i, item in enumerate(items)]

        distance_list = list(set([item[1][-1] for item in items]))
        distance_list.sort()

        if len(distance_list) > self.leaf_size:
            n_children = self.max_children
        else:
            n_children = len(distance_list)

        split_points = [-1]
        for i in range(n_children):
            split_points.append(distance_list[(i + 1) * (len(distance_list) - 1) // n_children])

        for i in range(n_children):
            child_items = [item for item in items if split_points[i] < item[1][-1] <= split_points[i + 1]]
            child = self._make_node(child_items)
            if child:
                node.children.append(child)

        return node

    def _get_middle_point_index(self, items):
        return 0

    def find(self, item):
        heap = [(0, 1, self.root, ())]

        while heap:
            top = heapq.heappop(heap)
            if top[1]:
                top[2].help_find(item, top[3], heap, self.dist_fn)
            else:
                yield top[2], top[0]


class CustomVPTree2(CustomVPTree):
    def __init__(self, dist: Dist, embedder: Callable):
        CustomVPTree.__init__(self, dist)
        self.embedder = embedder

    def _get_middle_point_index(self, items):
        points = [self.embedder(i[0]) for i in items]
        avg = np.average(points, axis=0)
        res = None
        res_d = 1e+9
        for i, p in enumerate(points):
            d = np.linalg.norm(p - avg)
            if d < res_d:
                res_d = d
                res = i
        return res


class CustomVPTree3(CustomVPTree):
    def _get_middle_point_index(self, points):
        if len(points) == 1:
            return 0

        res = None
        res_d = 0
        for i in range(min(len(points), 10)):
            ds = [self.dist_fn(points[i][0], p[0]) for p in points]
            d = np.sum(ds)
            if d > res_d:
                res_d = d
                res = i
        return res


class CustomVPTree4(CustomVPTree):
    def _get_middle_point_index(self, points):
        if len(points) == 1:
            return 0

        res = None
        res_d = 1e+9
        for i in range(min(len(points), 10)):
            ds = [self.dist_fn(points[i][0], p[0]) for p in points]
            d = np.sum(ds)
            if d < res_d:
                res_d = d
                res = i
        return res


class CustomVPTree5(CustomVPTree):
    def _get_middle_point_index(self, points):
        return random.randint(0, len(points) - 1)