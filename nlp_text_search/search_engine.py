from .custom_vp_tree import *
from deeppavlov import *
from .dists import *
import json
import os
import pickle
from typing import Any, List, Tuple, Type
from .vp_tree import VP_tree


class AbstractSearchEngine(metaclass=ABCMeta):
    def __init__(self, dist: Dist):
        self.dist = dist

    @abstractmethod
    def fit(self, points: List) -> None:
        pass

    @abstractmethod
    def search(self, point, n_neighbors) -> List[Tuple[Any, float]]:
        pass


class BruteForceSearchEngine(AbstractSearchEngine):
    def fit(self, points: List) -> None:
        self.points = points

    def search(self, point, n_neighbors) -> List[Tuple[Any, float]]:
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


class VPTree2SearchEngine(AbstractSearchEngine):
    def fit(self, points: List) -> None:
        self.tree = VP_tree(points, self.dist.dist)

    def search(self, point, n_neighbors) -> List[Tuple[Any, float]]:
        res = []
        i = 0
        for r in self.tree.find(point):
            res.append((r[0], r[1]))
            i += 1
            if i >= n_neighbors:
                break
        res.sort(key=lambda x: x[1])
        return res


class CustomVPTreeSearchEngine(VPTree2SearchEngine):
    def __init__(self, model_settings: dict, doc2vec: Doc2Vec, dist_class: Type[Dist2]):
        self.model_settings = model_settings
        self.model = build_model(model_settings, download=False)
        self.doc2vec = doc2vec
        VPTree2SearchEngine.__init__(self, dist_class(self.model, doc2vec))

    def set_tree(self, tree: CustomVPTree) -> None:
        self.tree = tree
        self.tree.set_dist(self.dist)

    def fit(self, points: List) -> None:
        self.tree = CustomVPTree(self.dist)
        self.tree.fit(points)

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        settings = {
            'self_class': type(self).__name__,
            'dist_class' : type(self.dist).__name__
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
            model_settings['metadata']['variables'] = dict([('SE_PATH', path)] + list(model_settings['metadata']['variables'].items()))

        doc2vec = Doc2Vec.load(os.path.join(path, 'doc2vec.model'))

        self_class = globals()[settings['self_class']]
        dist_class = globals()[settings['dist_class']]

        res: CustomVPTreeSearchEngine = self_class(model_settings, doc2vec, dist_class)

        with open(os.path.join(path, 'tree.model'), 'rb') as f:
            tree: CustomVPTree = pickle.load(f)
            res.set_tree(tree)

        return res
