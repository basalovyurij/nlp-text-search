"""
Based on http://www.logarithmic.net/pfh/blog/01164790008
"""
import deprecation
import heapq

from ..dists import Dist
from .._version import __version__


class _Node:
    def __init__(self):
        self.lower_bounds = []
        self.upper_bounds = []
        self.children = []
        self.vantage = None

    def minimum_distance(self, distances):
        minimum = 0.0
        for i in range(len(distances)):
            if distances[i] < self.lower_bounds[i]:
                minimum = max(minimum, self.lower_bounds[i] - distances[i])
            elif distances[i] > self.upper_bounds[i]:
                minimum = max(minimum, distances[i] - self.upper_bounds[i])
        return minimum

    def help_find(self, item, distances, heap, distance):
        d = distance(self.vantage, item)
        new_distances = distances + (d,)

        heapq.heappush(heap, (d, 0, self.vantage))

        for child in self.children:
            heapq.heappush(heap, (child.minimum_distance(new_distances), 1, child, new_distances))

    def __lt__(self, other):
        return len(self.children) < len(other.children)

    def __gt__(self, other):
        return len(self.children) > len(other.children)


class VP_tree(object):
    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use jvptree.VPTree instead")
    def __init__(self, distance, max_children=2):
        """ items        : list of items to make tree out of
            distance     : function that returns the distance between two items
            max_children : maximum number of children for each node

            Using larger max_children will reduce the time needed to construct the tree,
            but may make queries less efficient.
        """
        self.distance = distance
        self.max_children = max_children

    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use jvptree.VPTree instead")
    def fit(self, points) -> None:
        self.root = self._make_node([(item, ()) for item in points])

    def _make_node(self, items):
        if not items:
            return None

        node = _Node()
        for i in range(len(items[0][1])):
            distance_list = [item[1][i] for item in items]
            node.lower_bounds.append(min(distance_list))
            node.upper_bounds.append(max(distance_list))

        item = items.pop(0)
        node.vantage = item[0]

        if not items:
            return node

        items = self._fill_items(node, items)

        distances = {}
        for item in items:
            distances[item[1][-1]] = True
        distance_list = list(distances.keys())
        distance_list.sort()
        n_children = self._get_n_children(distance_list)
        split_points = [-1]
        for i in range(n_children):
            split_points.append(distance_list[(i + 1) * (len(distance_list) - 1) // n_children])

        for i in range(n_children):
            child_items = [item for item in items if split_points[i] < item[1][-1] <= split_points[i + 1]]
            child = self._make_node(child_items)
            if child:
                node.children.append(child)

        return node

    def _fill_items(self, node, items):
        return [(item[0], item[1] + (self.distance(node.vantage, item[0]),)) for item in items]

    def _get_n_children(self, distance_list):
        return min(self.max_children, len(distance_list))

    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use jvptree.VPTree instead")
    def find(self, item):
        """ Return iterator yielding items in tree in order of distance from supplied item.
        """
        if not self.root:
            return

        heap = [(0, 1, self.root, ())]

        while heap:
            top = heapq.heappop(heap)
            if top[1]:
                top[2].help_find(item, top[3], heap, self.distance)
            else:
                yield top[2], top[0]


class CustomVPTree(VP_tree):
    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use jvptree.VPTree instead")
    def __init__(self, dist: Dist):
        super().__init__(dist.dist)
        self.dist_batch_fn = dist.batch_dist
        self.max_children = 2
        self.leaf_size = 5

    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use jvptree.VPTree instead")
    def set_dist(self, dist: Dist):
        self.distance = dist.dist
        self.dist_batch_fn = dist.batch_dist

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.root, self.max_children, self.leaf_size

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.root, self.max_children, self.leaf_size = state

    def _fill_items(self, node, items):
        dists = self.dist_batch_fn([(node.vantage, item[0]) for item in items])
        return [(item[0], item[1] + (dists[i],)) for i, item in enumerate(items)]

    def _get_n_children(self, distance_list):
        if len(distance_list) > self.leaf_size:
            return self.max_children
        else:
            return len(distance_list)