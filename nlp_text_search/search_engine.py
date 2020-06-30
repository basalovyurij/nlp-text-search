from abc import ABCMeta, abstractmethod
from deeppavlov import build_model
import deprecation
from gensim.models import Doc2Vec
import json
import os
import pickle
from typing import Any, List, Tuple, Type

from .dists import Dist, LinearizedDist
from .vptree.jvptree import VPTree
from .vptree.simple_vptree import VP_tree, CustomVPTree
from ._version import __version__


class AbstractSearchEngine(metaclass=ABCMeta):
    def __init__(self, dist: Dist):
        self.dist = dist

    @abstractmethod
    def fit(self, points: List[Any]) -> None:
        pass

    @abstractmethod
    def search(self, point: Any, n_neighbors: int = 10) -> List[Tuple[Any, float]]:
        pass


class BruteForceSearchEngine(AbstractSearchEngine):
    def fit(self, points: List[Any]) -> None:
        self.points = points

    def search(self, point: Any, n_neighbors: int = 10) -> List[Tuple[Any, float]]:
        res = []
        md = 0
        for p in self.points:
            d = self.dist.dist(p, point)
            if len(res) < n_neighbors:
                res.append((p, d))
                md = max(md, d)
            elif d < md:
                res.append((p, d))
                res.sort(key=lambda x: x[1])
                res.pop()
        return res


class VPTreeSearchEngine(AbstractSearchEngine):
    def fit(self, points: List[Any]) -> None:
        self.tree = VP_tree(self.dist.dist)
        self.tree.fit(points)

    def search(self, point: Any, n_neighbors: int = 10) -> List[Tuple[Any, float]]:
        res = []
        i = 0
        for r in self.tree.find(point):
            res.append((r[0], r[1]))
            i += 1
            if i >= n_neighbors:
                break
        res.sort(key=lambda x: x[1])
        return res


class SaveableVPTreeSearchEngine(VPTreeSearchEngine):
    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use DefaultSearchEngine instead")
    def __init__(self, model_settings: dict, doc2vec: Doc2Vec,
                 dist_class: Type[LinearizedDist] = Dist, linearization_settings: dict = {}):
        self.model_settings = model_settings
        self.model = build_model(model_settings, download=True)
        self.doc2vec = doc2vec
        self.dist = dist_class(self.model, self.doc2vec, linearization_settings)
        VPTreeSearchEngine.__init__(self, self.dist)

    def _set_tree(self, tree: CustomVPTree) -> None:
        self.tree = tree
        self.tree.set_dist(self.dist)

    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use DefaultSearchEngine instead")
    def fit(self, points: List[Any]) -> None:
        self.tree = CustomVPTree(self.dist)
        self.tree.fit(points)

    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use DefaultSearchEngine instead")
    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        settings = {
            'self_class': type(self).__name__,
            'dist_class': type(self.dist).__name__,
            'linearization_settings': self.dist.get_linearization_settings()
        }

        with open(os.path.join(path, 'settings.json'), 'w') as f:
            json.dump(settings, f, indent=True)

        with open(os.path.join(path, 'model_settings.json'), 'w') as f:
            json.dump(self.model_settings, f, indent=True)

        Doc2Vec.save(self.doc2vec, os.path.join(path, 'doc2vec.model'))

        with open(os.path.join(path, 'tree.model'), 'wb') as f:
            pickle.dump(self.tree, f)

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.6", removed_in="1.0", current_version=__version__,
                            details="Use DefaultSearchEngine instead")
    def load(path):
        with open(os.path.join(path, 'settings.json'), 'r') as f:
            settings = json.load(f)

        with open(os.path.join(path, 'model_settings.json'), 'r') as f:
            model_settings = json.load(f)
            # model_settings['metadata']['variables'] = dict([('SE_PATH', path)] + list(model_settings['metadata']['variables'].items()))

        doc2vec = Doc2Vec.load(os.path.join(path, 'doc2vec.model'))

        self_class = globals()[settings['self_class']]
        dist_class = globals()[settings['dist_class']]
        linearization_settings = settings['linearization_settings']

        res: SaveableVPTreeSearchEngine = self_class(model_settings, doc2vec, dist_class, linearization_settings)

        with open(os.path.join(path, 'tree.model'), 'rb') as f:
            tree: CustomVPTree = pickle.load(f)
            res._set_tree(tree)

        return res


class DefaultSearchEngine(AbstractSearchEngine):
    def __init__(self, model_settings: dict, doc2vec: Doc2Vec,
                 dist_class: Type[LinearizedDist] = Dist, linearization_settings: dict = {},
                 points: List[Any] = []):
        self.model_settings = model_settings
        self.model = build_model(model_settings, download=True)
        self.doc2vec = doc2vec
        self.dist = dist_class(self.model, self.doc2vec, linearization_settings)
        self.tree = VPTree(self.dist, points)
        AbstractSearchEngine.__init__(self, self.dist)

    def _set_tree(self, tree: VPTree) -> None:
        self.tree = tree
        self.tree.set_dist(self.dist)

    def search(self, point: Any, n_neighbors: int = 10) -> List[Tuple[Any, float]]:
        res = self.tree.get_nearest_neighbors(point, n_neighbors)
        return res

    def fit(self, points: List[Any]) -> None:
        self.tree = VPTree(self.dist, points)

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        settings = {
            'self_class': type(self).__name__,
            'dist_class': type(self.dist).__name__,
            'linearization_settings': self.dist.get_linearization_settings()
        }

        with open(os.path.join(path, 'settings.json'), 'w') as f:
            json.dump(settings, f, indent=True)

        with open(os.path.join(path, 'model_settings.json'), 'w') as f:
            json.dump(self.model_settings, f, indent=True)

        Doc2Vec.save(self.doc2vec, os.path.join(path, 'doc2vec.model'))

        with open(os.path.join(path, 'tree.model'), 'wb') as f:
            pickle.dump(self.tree, f)

    @staticmethod
    def load(path):
        with open(os.path.join(path, 'settings.json'), 'r') as f:
            settings = json.load(f)

        with open(os.path.join(path, 'model_settings.json'), 'r') as f:
            model_settings = json.load(f)
            #  model_settings['metadata']['variables'] = dict([('SE_PATH', path)] + list(model_settings['metadata']['variables'].items()))

        doc2vec = Doc2Vec.load(os.path.join(path, 'doc2vec.model'))

        self_class = globals()[settings['self_class']]
        dist_class = globals()[settings['dist_class']]
        linearization_settings = settings['linearization_settings']

        res: DefaultSearchEngine = self_class(model_settings, doc2vec, dist_class, linearization_settings)

        with open(os.path.join(path, 'tree.model'), 'rb') as f:
            tree: VPTree = pickle.load(f)
            res._set_tree(tree)

        return res
