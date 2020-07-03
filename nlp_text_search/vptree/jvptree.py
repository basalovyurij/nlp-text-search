"""
Based on https://github.com/jchambers/jvptree
"""
import heapq
import random
from typing import Any, List, Tuple, Union

from ..dists import Dist


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


class ThresholdSelectionStrategy(object):
    """
    A strategy for choosing a distance threshold for vp-tree nodes. The main feature of vp-trees is that they partition
    collections of points into collections of points that are closer to a given point (the vantage point) than a certain
    threshold or farther away from the vantage point than the threshold. Given a list of points, a
    ThresholdSelectionStrategy chooses the distance that will be used by a vp-tree node to partition its points.
    """
    def select_threshold(self, points: List[Any], origin: Any, dist: Dist) -> float:
        """
        Chooses a partitioning distance threshold appropriate for the given list of points.
        Implementations are allowed to reorder the list of points, but must not add or remove points from the list.

        :param points: the points for which to choose a partitioning distance threshold
        :param origin: the point from which the threshold distances should be calculated
        :param dist: the function to be used to calculate distances between points
        :return: a partitioning threshold distance appropriate for the given list of points;
            ideally, some points should be closer to the origin than the returned threshold, and some should be farther
        """
        raise NotImplementedError()


class MedianDistanceThresholdSelectionStrategy(ThresholdSelectionStrategy):
    """
    A threshold distance selection strategy that uses the median distance from the origin as the threshold.
    """
    def select_threshold(self, points: List[Any], origin: Any, dist: Dist) -> float:
        """
        Returns the median distance of the given points from the given origin.
        This method will partially sort the list of points in the process.

        :param points: the points for which to choose a partitioning distance threshold
        :param origin: the point from which the threshold distances should be calculated
        :param dist: the function to be used to calculate distances between points
        :return: the median distance from the origin to the given list of points
        """
        dists = dist.batch_dist([(origin, point) for point in points])

        left = 0
        right = len(points) - 1
        median_index = len(points) // 2
        if dists[median_index] == max(dists):
            median_index += 1

        # The strategy here is to use quickselect (https://en.wikipedia.org/wiki/Quickselect) to recursively partition
        # the parts of a list on one side of a pivot, working our way toward the center of the list.
        while left != right:
            pivot_index = left + random.randrange(right - left)
            pivot_distance = dists[pivot_index]

            # Temporarily move the pivot point all the way out to the end of this section of the list
            swap(points, pivot_index, right)
            swap(dists, pivot_index, right)

            store_index = left

            for i in range(left, right):
                if dists[i] < pivot_distance:
                    swap(points, store_index, i)
                    swap(dists, store_index, i)
                    store_index += 1

            # ...and now bring that original pivot point back to its rightful place.
            swap(points, store_index, right)
            swap(dists, store_index, right)

            if store_index == median_index:
                break  # Mission accomplished; we've placed the point that should rightfully be at the median index
            elif store_index < median_index:
                left = store_index + 1  # We need to work on the section of the list to the right of the pivot
            else:
                right = store_index - 1  # We need to work on the section of the list to the left of the pivot

        return dists[median_index]


class SamplingMedianDistanceThresholdSelectionStrategy(MedianDistanceThresholdSelectionStrategy):
    """
    A threshold distance selection strategy that uses the median distance from the origin to a subset
    of the given list of points as the threshold.
    """
    def __init__(self, number_of_samples: int = 32):
        self.number_of_samples = number_of_samples

    def select_threshold(self, points: List[Any], origin: Any, dist: Dist) -> float:
        """
        Returns the median distance of a subset of the given points from the given origin.
        The given list of points may be partially sorted in the process.

        :param points: the points for which to choose a partitioning distance threshold
        :param origin: the point from which the threshold distances should be calculated
        :param dist: the function to be used to calculate distances between points
        :return: the median distance from the origin to the given list of points
        """
        return MedianDistanceThresholdSelectionStrategy.select_threshold(self, self._get_sampled_points(points),
                                                                         origin, dist)

    def _get_sampled_points(self, points: List[Any]) -> List[Any]:
        if len(points) <= self.number_of_samples:
            return points

        step = len(points) // self.number_of_samples
        sampled_points = [points[i * step] for i in range(self.number_of_samples)]
        return sampled_points


class NearestNeighborCollector(object):
    """
    A utility class that uses a priority queue to efficiently collect results
    for a k-nearest-neighbors query in a vp-tree.
    """
    def __init__(self, query_point: Any, dist: Dist, capacity: int):
        """
        Constructs a new nearest neighbor collector that selectively accepts points that are close to the given query
        point as determined by the given distance function. Up to the given number of nearest neighbors are collected,
        and if neighbors are found that are closer than points in the current set, the most distant previously collected
        point is replaced with the closer candidate.

        :param query_point: the point for which nearest neighbors are to be collected
        :param dist: the distance function to be used to determine the distance between
            the query point and potential neighbors
        :param capacity: the maximum number of nearest neighbors to collect
        """
        self.query_point = query_point
        self.dist = dist
        self.capacity = capacity
        self.priority_queue: List[Tuple[Any, float]] = []
        self.distance_to_farthest_point = 0

    def offer_points(self, points: List[Any]) -> None:
        """
        Offers a point to this collector. The point may or may not be added to the collection; points will only be added
        if the collector is not already full, or if the collector is full, but the offered point is closer to the query
        point than the most distant point already in the collection.

        :param points: the points to offer to this collector
        """
        point_added = False
        distances_to_new_points = self.dist.batch_dist([(self.query_point, point) for point in points])
        for i, point in enumerate(points):
            queue_item = (-distances_to_new_points[i], point)
            if len(self.priority_queue) < self.capacity:
                heapq.heappush(self.priority_queue, queue_item)
                point_added = True
            else:
                if distances_to_new_points[i] < self.distance_to_farthest_point:
                    heapq.heappushpop(self.priority_queue, queue_item)
                    point_added = True

            if point_added:
                self.distance_to_farthest_point = -self.priority_queue[0][0]

    def get_farthest_point(self) -> Any:
        """
        Returns the point retained by this collector that is the farthest from the query point.
        :return: the point retained by this collector that is the farthest from the query point
        """
        return self.priority_queue[0][1]

    def to_sorted_list(self) -> List[Tuple[Any, float]]:
        """
        Returns a list of points retained by this collector, sorted by distance from the query point.

        :return: a list of points retained by this collector, sorted by distance from the query point
        """
        return sorted([(i[1], -i[0]) for i in self.priority_queue], key=lambda i: i[1])


class VPTreeNode(object):
    """
    A single node of a vantage-point tree. Nodes may either be leaf nodes that contain points directly or branch nodes
    that have a "closer than threshold" and "farther than threshold" child node.
    """
    def __init__(self, points: List[Any], dist: Dist,
                 threshold_selection_strategy: ThresholdSelectionStrategy, capacity: int):
        """
        Constructs a new node that contains the given collection of points. If the given collection of points is larger
        than the given maximum capacity, the new node will attempts to partition the collection of points into child
        nodes using the given distance function and threshold selection strategy.

        :param points: the collection of points to store in or below this node
        :param dist: the distance function to use when partitioning points
        :param threshold_selection_strategy: the threshold selection strategy to use when selecting points
        :param capacity: the desired maximum capacity of this node; this node may contain more points than the given
            capacity if the given collection of points cannot be partitioned (for example, because all of the points
            are an equal distance away from the vantage point)
        """
        self.capacity = capacity
        self.dist = dist
        self.threshold_selection_strategy = threshold_selection_strategy
        self.points: Union[List[Any], None] = points.copy()
        self.threshold: Union[float, None] = None
        self.closer: Union[VPTreeNode, None] = None
        self.farther: Union[VPTreeNode, None] = None

        # All nodes must have a vantage point; choose one at random from the available points
        self.vantage_point = points[random.randrange(len(points))]

        self.anneal()

    def set_tree_data(self, dist: Dist, threshold_selection_strategy: ThresholdSelectionStrategy,
                      capacity: int) -> None:
        """
        Recursively set tree params to node loaded from pickle
        :param dist: the VPTree distance function to use when partitioning points
        :param threshold_selection_strategy: the VPTree threshold selection strategy to use when selecting points
        :param capacity: the VPTree desired maximum capacity of nodes
        :return:
        """
        self.dist = dist
        self.threshold_selection_strategy = threshold_selection_strategy
        self.capacity = capacity

        if self.closer is not None:
            self.closer.set_tree_data(dist, threshold_selection_strategy, capacity)

        if self.farther is not None:
            self.farther.set_tree_data(dist, threshold_selection_strategy, capacity)

    def __getstate__(self) -> Tuple[Union[List[Any], None], Union[float, None], Any,
                                    Union['VPTreeNode', None], Union['VPTreeNode', None]]:
        """Return state values to be pickled."""
        return self.points, self.threshold, self.vantage_point, self.closer, self.farther

    def __setstate__(self, state: Tuple[Union[List[Any], None], Union[float, None], Any,
                                        Union['VPTreeNode', None], Union['VPTreeNode', None]]) -> None:
        """
        Restore state from the unpickled state values.
        :param state: state created using __getstate__ function
        """
        self.points, self.threshold, self.vantage_point, self.closer, self.farther = state

    def __len__(self) -> int:
        """
        Returns the number of points stored in this node and its children.

        :return: Returns the number of points stored in this node and its children.
        """
        if self.points is None:
            return len(self.closer) + len(self.farther)
        else:
            return len(self.points)

    def _add_all_points_to_collection(self, collection: List[Any]) -> None:
        """
        Adds all points contained by this node and its children to the given collection.

        :param collection: the collection to which points should be added.
        """
        if self.points is None:
            self.closer._add_all_points_to_collection(collection)
            self.farther._add_all_points_to_collection(collection)
        else:
            collection.extend(self.points)

    def _partition_points(self) -> int:
        """
        Partitions the points in the given list such that all points that fall within the given distance threshold of
        the given vantage point are on one "side" of the list and all points beyond the threshold are on the other.

        :return: the index of the first point in the list that falls beyond the distance threshold or -1 if the list of
            points could not be partitioned (i.e. because they are all the same distance from the vantage point
        """
        left = 0
        right = len(self.points) - 1
        dists = self.dist.batch_dist([(self.vantage_point, point) for point in self.points])

        # This is, essentially, a single swapping quicksort iteration
        while left <= right:
            if dists[left] > self.threshold:
                while right >= left:
                    if dists[right] <= self.threshold:
                        swap(self.points, left, right)
                        swap(dists, left, right)
                        right -= 1
                        break
                    right -= 1
            left += 1

        if dists[0] <= self.threshold < dists[-1]:
            first_index_past_threshold = left - 1 if dists[left - 1] > self.threshold else left
            return first_index_past_threshold

        return -1

    def anneal(self) -> None:
        """
        Rebuild VPTreeNode
        """
        if self.points is None:
            closer_size = len(self.closer)
            farther_size = len(self.farther)

            if closer_size == 0 or farther_size == 0:
                # One of the child nodes has become empty, and needs to be pruned.
                self.points = []
                self._add_all_points_to_collection(self.points)
                self.closer = None
                self.farther = None
                self.anneal()
            else:
                self.closer.anneal()
                self.farther.anneal()
        elif len(self.points) > self.capacity:
            # Partially sort the list such that all points closer than or equal to the threshold distance from the
            # vantage point come before the threshold point in the list and all points farther away come after the
            # threshold point.
            self.threshold = self.threshold_selection_strategy.select_threshold(self.points, self.vantage_point,
                                                                                self.dist)
            first_index_past_threshold = self._partition_points()
            if first_index_past_threshold == -1:
                # We couldn't partition the list, so just store all of the points in this node
                self.closer = None
                self.farther = None
            else:
                self.closer = VPTreeNode(self.points[:first_index_past_threshold], self.dist,
                                         self.threshold_selection_strategy, self.capacity)
                self.farther = VPTreeNode(self.points[first_index_past_threshold:], self.dist,
                                          self.threshold_selection_strategy, self.capacity)
                self.points = None

    def get_child_node_for_point(self, point: any) -> 'VPTreeNode':
        """
        Returns the child node (either the closer node or farther node) that would contain the given point given its
        distance from this node's vantage point.

        :param point: the point for which to choose an appropriate child node; the point need not actually exist
            within either child node
        :return: this node's "closer" child node if the given point is within this node's distance threshold
            of the vantage point or the "farther" node otherwise
        """
        if self.dist.dist(self.vantage_point, point) <= self.threshold:
            return self.closer
        else:
            return self.farther

    def add(self, point: Any) -> None:
        """
        Adds a point to this node or one of its children. If this node is a leaf node and the addition of the new point
        increases the size of the node beyond its desired capacity, the node will attempt to partition its points into
        two child nodes.

        :param point: the point to add to this node
        """
        if self.points is None:
            # This is not a leaf node; pass this point on to the appropriate child
            self.get_child_node_for_point(point).add(point)
        else:
            self.points.append(point)

    def remove(self, point: Any) -> None:
        """
        Removes a point from this node (if it is a leaf node) or one of its children. If the removal of the point would
        result in an empty node, the empty node's parent will absorb and re-partition all points from all child nodes.

        :param point: the point to remove from this node or one of its children
        """
        if self.points is None:
            # This is not a leaf node; try to remove the point from an appropriate child node
            self.get_child_node_for_point(point).remove(point)
        else:
            self.points.remove(point)

    def collect_nearest_neighbors(self, collector: NearestNeighborCollector):
        if self.points is not None:
            collector.offer_points(self.points)
            return

        first_node_searched = self.get_child_node_for_point(collector.query_point)
        first_node_searched.collect_nearest_neighbors(collector)

        dists = self.dist.batch_dist([
            (self.vantage_point, collector.query_point),
            (collector.query_point, collector.get_farthest_point())])
        dist_from_vantage_point_to_query_point = dists[0]
        dist_from_query_point_to_farthest_point = dists[1]

        if first_node_searched == self.closer:
            # We've already searched the node that contains points within this node's threshold. We also want to
            # search the farther node if the distance from the query point to the most distant point in the
            # neighbor collector is greater than the distance from the query point to this node's threshold, since
            # there could be a point outside of this node that's closer than the most distant neighbor we've found
            # so far.
            if self.threshold - dist_from_vantage_point_to_query_point < dist_from_query_point_to_farthest_point:
                self.farther.collect_nearest_neighbors(collector)
        else:
            # We've already searched the node that contains points beyond this node's threshold. We want to search
            # the within-threshold node if it's "easier" to get from the query point to this node's region than it
            # is to get from the query point to the most distant match, since there could be a point within this
            # node's threshold that's closer than the most distant match.
            if dist_from_vantage_point_to_query_point - self.threshold <= dist_from_query_point_to_farthest_point:
                self.closer.collect_nearest_neighbors(collector)


class VPTree(object):
    """
    A vantage-point tree (or vp-tree) is a binary space partitioning collection of points in a metric space. The main
    feature of vantage point trees is that they allow for k-nearest-neighbor searches in any metric space in
    **O(log(n))** time.

    Vantage point trees recursively partition points by choosing a &quot;vantage point&quot; and a distance threshold;
    points are then partitioned into one collection that contains all of the points closer to the vantage point than the
    chosen threshold and one collection that contains all of the points farther away than the chosen threshold.

    A distance function that satisfies the properties of a metric space must be provided
    when constructing a vantage point tree. Callers may also specify a threshold selection strategy (a sampling median
    strategy is used by default) and a node size to tune the ratio of nodes searched to points inspected per node.
    Vantage point trees may be constructed with or without an initial collection of points, though specifying a
    collection of points at construction time is the most efficient approach.
    """
    def __init__(self, dist: Dist, points: Union[List[Any], None] = None,
                 threshold_selection_strategy: Union[ThresholdSelectionStrategy, None] = None, node_capacity: int = 32):
        """
        Constructs a new vp-tree that uses the given distance function and threshold selection strategy to partition
        points. The tree will attempt to partition nodes that contain more than *node_capacity* points, and will
        be initially populated with the given collection of points.

        :param dist: the distance function to use to calculate the distance between points
        :param points: the points with which this tree should be initially populated; may be None
        :param threshold_selection_strategy: the function to use to choose distance thresholds when partitioning nodes
        :param node_capacity: the largest capacity a node may have before it should be partitioned
        """
        self.dist = dist
        if threshold_selection_strategy is None:
            threshold_selection_strategy = SamplingMedianDistanceThresholdSelectionStrategy()
        self.threshold_selection_strategy = threshold_selection_strategy
        self.node_capacity = node_capacity

        self.root_node: Union[VPTreeNode, None] = None
        if points:
            self.root_node = VPTreeNode(points, self.dist, self.threshold_selection_strategy, self.node_capacity)

    def set_dist(self, dist: Dist) -> None:
        """
        Set dist to loaded pickle tree
        :param dist: builded Dist func
        """
        if hasattr(self, 'dist'):
            raise ValueError('Dist function already set for VPTree')
        self.dist = dist

        if not hasattr(self, 'threshold_selection_strategy') or not hasattr(self, 'node_capacity'):
            raise ValueError('VPTree not initialized. Call __init__() or load pickle model before calling set_dist()')
        self.root_node.set_tree_data(dist, self.threshold_selection_strategy, self.node_capacity)

    def __getstate__(self) -> Tuple[VPTreeNode, ThresholdSelectionStrategy, int]:
        """Return state values to be pickled."""
        return self.root_node, self.threshold_selection_strategy, self.node_capacity

    def __setstate__(self, state: Tuple[VPTreeNode, ThresholdSelectionStrategy, int]) -> None:
        """
        Restore state from the unpickled state values.
        :param state: state created using __getstate__ function
        """
        self.root_node, self.threshold_selection_strategy, self.node_capacity = state

    def _check_initialized(self):
        for a in ['dist', 'threshold_selection_strategy', 'node_capacity']:
            if not hasattr(self, a):
                raise ValueError('VPTree not initialized. Call __init__() or load pickle model and call set_dist()')

    def get_nearest_neighbors(self, query_point: Any, max_results: int = 10) -> Union[List[Tuple[Any, float]], None]:
        """
        Returns a list of the nearest neighbors to a given query point. The returned list is sorted by increasing
        distance from the query point.

        This returned list will contain at most *max_results* elements (and may contain fewer if
        *max_results* is larger than the number of points in the index). If multiple points have the same distance
        from the query point, the order in which they appear in the returned list is undefined. By extension,
        if multiple points have the same distance from the query point and those points would "straddle" the end of the
        returned list, which points are included in the list and which are cut off is not prescribed.

        :param query_point: the point for which to find neighbors
        :param max_results: the maximum length of the returned list
        :return: a list of the nearest neighbors to the given query point sorted by increasing distance
            from the query point
        """
        self._check_initialized()

        if self.root_node is None:
            return None

        collector = NearestNeighborCollector(query_point, self.dist, max_results)
        self.root_node.collect_nearest_neighbors(collector)
        nearest_neighbors = collector.to_sorted_list()
        return nearest_neighbors

    def add(self, point: Any) -> None:
        """
        Add point to existing tree
        :param point: the point to add
        """
        self.add_all([point])

    def add_all(self, points: List[Any]) -> None:
        """
        Add multiple points to existing tree
        :param points: the points to add
        """
        self._check_initialized()

        if self.root_node is None:
            # We don't need to anneal here because annealing happens automatically as part of node construction
            self.root_node = VPTreeNode(points, self.dist, self.threshold_selection_strategy, self.node_capacity)
        else:
            for point in points:
                self.root_node.add(point)
            if points:
                self.root_node.anneal()

    def remove(self, point: Any) -> None:
        """
        Remove point to existing tree
        :param point: the point to remove
        """
        self.remove_all([point])

    def remove_all(self, points: List[Any]) -> None:
        """
        Remove multiple points to existing tree
        :param points: the points to remove
        """
        self._check_initialized()

        if self.root_node is not None:
            for point in points:
                self.root_node.remove(point)
            if points:
                self.root_node.anneal()
