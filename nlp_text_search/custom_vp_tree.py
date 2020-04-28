from .dists import Dist
from .vp_tree import *


class CustomVPTree(VP_tree):
    def __init__(self, dist: Dist):
        super().__init__(dist.dist)
        self.dist_batch_fn = dist.batch_dist
        self.max_children = 2
        self.leaf_size = 5

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