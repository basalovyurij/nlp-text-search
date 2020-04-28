from abc import ABCMeta, abstractmethod
from deeppavlov.core.common.chainer import Chainer
from gensim.models import Doc2Vec
from lru import LRU
from methodtools import lru_cache
import nltk
import numpy as np
from typing import List, Tuple


class Stat:
    def __init__(self):
        self.dist_calls_count = 0


stat = Stat()


class Dist(metaclass=ABCMeta):
    @abstractmethod
    def dist(self, x: str, y: str) -> float:
        pass

    @abstractmethod
    def batch_dist(self, X: List[Tuple[str, str]]) -> List[float]:
        pass


class LinearizedDist(Dist):
    def __init__(self, model: Chainer, doc2vec: Doc2Vec, linearization_settings: dict):
        self.model = model
        self.doc2vec = doc2vec
        self.scale_edge = linearization_settings.get('scale_edge', 0.5)
        self.scale_coef = linearization_settings.get('scale_coef', 0.5)
        self.doc2vec_weight = linearization_settings.get('scale_coef', 0.1)
        self.model_weight = linearization_settings.get('scale_coef', 1)
        self.cache = LRU(1000000)

    def get_linearization_settings(self):
        return {
            'scale_edge': self.scale_edge,
            'scale_coef': self.scale_coef,
            'doc2vec_weight': self.doc2vec_weight,
            'model_weight': self.model_weight
        }

    @lru_cache(maxsize=1000000)
    def _embed(self, s):
        w = nltk.word_tokenize(s, preserve_line=True)
        return self.doc2vec.infer_vector(w)

    def _scalar(self, x, y):
        v1 = self._embed(x)
        v2 = self._embed(y)
        return np.arccos(np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))) / np.pi

    def _combine(self, sc, ds1, ds2):
        # weighting
        d = self.doc2vec_weight * sc + self.model_weight * (2 - (ds1 + ds2))
        # normalizing
        d = d / (self.doc2vec_weight + 2 * self.model_weight)
        #return d
        # linearizing distribution
        if d < self.scale_edge:
            return d * self.scale_coef / self.scale_edge
        else:
            return (1 - self.scale_coef) * (d - self.scale_edge) / (1 - self.scale_edge) + self.scale_coef

    def _get_from_cache(self, x: str, y: str):
        stat.dist_calls_count += 1

        if x == y:
            return 0

        k = (x, y)
        if k in self.cache:
            return self.cache[k]

        k2 = (y, x)
        if k2 in self.cache:
            return self.cache[k2]

        return None

    def dist(self, x: str, y: str) -> float:
        d = self._get_from_cache(x, y)
        if d is None:
            sc = np.round(self._scalar(x, y), decimals=5)
            ds = np.round(self.model([[x, y], [y, x]]), decimals=5)
            d = self._combine(sc, ds[0], ds[1])
            self.cache[(x, y)] = d

        return d

    def batch_dist(self, X: List[Tuple[str, str]]) -> List[float]:
        res = [0] * len(X)
        new_pairs = []
        Y = []
        for i, (x, y) in enumerate(X):
            d = self._get_from_cache(x, y)
            if d is not None:
                res[i] = d
            else:
                new_pairs.append(i)
                Y.append((x, y))

        sc = np.round([self._scalar(x, y) for (x, y) in Y], decimals=5)
        ds = np.round(self.model([[x, y] for (x, y) in Y] + [[y, x] for (x, y) in Y]), decimals=5)

        l = len(new_pairs)
        for i, (x, y) in enumerate(Y):
            d = self._combine(sc[i], ds[i], ds[l + i])
            self.cache[(x, y)] = d
            res[new_pairs[i]] = d

        return res
