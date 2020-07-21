import math
from nlp_text_search.dists import Dist
from nlp_text_search.vptree.jvptree import VPTreeNode, SamplingSortDistanceThresholdSelectionStrategy, RandomVantagePointSelectionStrategy
import numpy as np
from typing import Any, List, Tuple
from unittest import TestCase


class TestDist(Dist):
    def dist(self, x: np.ndarray, y: np.ndarray):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    def batch_dist(self, x: List[Tuple[Any, Any]]) -> List[float]:
        return [self.dist(a, b) for a, b in x]


class TestVPTreeNodeCreation(TestCase):
    def __init__(self):
        TestCase.__init__(self, 'test')
        self.node_capacity = 10

    def test(self):
        for i in range(10):
            points = np.random.random((10000, 2)).tolist()
            node = VPTreeNode(points, TestDist(), SamplingSortDistanceThresholdSelectionStrategy(),
                              RandomVantagePointSelectionStrategy(), self.node_capacity)
            self._check(node)

    def _check(self, node: VPTreeNode):
        self.assertTrue(node.points is None or len(node.points) <= self.node_capacity)
        if node.farther is not None:
            self._check(node.farther)
        if node.closer is not None:
            self._check(node.closer)
