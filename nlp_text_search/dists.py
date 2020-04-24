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


class Dist1(Dist):
    def __init__(self, model: Chainer):
        self.model = model

    def _jaccard(self, seq1, seq2):
        set1, set2 = set(seq1), set(seq2)
        return len(set1 & set2) / float(len(set1 | set2))

    def dist(self, x: str, y: str) -> float:
        stat.dist_calls_count += 1
        if x == y:
            return 0

        d = self.model([[x, y], [y, x]])

        w1 = nltk.word_tokenize(x, preserve_line=True)
        w2 = nltk.word_tokenize(y, preserve_line=True)
        jc = self._jaccard(w1, w2)
        return 0.001 * (1 - jc) + 0.999 * (1 - np.log2(max(d[0] + d[1], 10 ** -7)))

    def batch_dist(self, X: List[Tuple[str, str]]) -> List[float]:
        stat.dist_calls_count += len(X)

        ds = self.model([[i[0], i[1]] for i in X] + [[i[1], i[0]] for i in X])

        res = []
        l = len(X)
        for i in range(l):
            d = 0
            if X[i][0] != X[i][1]:
                w1 = nltk.word_tokenize(X[i][0], preserve_line=True)
                w2 = nltk.word_tokenize(X[i][1], preserve_line=True)
                jc = self._jaccard(w1, w2)
                d = 0.001 * (1 - jc) + 0.999 * (1 - np.log2(max(ds[i] + ds[l + i], 10 ** -7)))
            res.append(d)

        return res


class Dist2(Dist):
    def __init__(self, model: Chainer, doc2vec: Doc2Vec):
        self.model = model
        self.doc2vec = doc2vec
        self.doc2vec_weight = 1
        self.model_weight = 1
        self.zero_point = np.log2(self.doc2vec_weight + self.model_weight * 2)

    @lru_cache(maxsize=1000000)
    def _embed(self, s):
        w = nltk.word_tokenize(s, preserve_line=True)
        return self.doc2vec.infer_vector(w)

    def _scalar(self, x, y):
        v1 = self._embed(x)
        v2 = self._embed(y)
        return np.dot(v1, v2)

    def dist(self, x: str, y: str) -> float:
        stat.dist_calls_count += 1

        if x == y:
            return 0

        sc = np.round(np.abs(self._scalar(x, y)), decimals=5)
        ds = np.round(self.model([[x, y], [y, x]]), decimals=5)
        d = np.round(self.doc2vec_weight * sc + self.model_weight * (ds[0] + ds[1]), decimals=5)

        return self.zero_point - np.log2(max(d, 10 ** -6))

    def batch_dist(self, X: List[Tuple[str, str]]) -> List[float]:
        stat.dist_calls_count += len(X)

        l = len(X)
        sc = np.round(np.abs([self._scalar(X[i][0], X[i][1]) for i in range(l)]), decimals=5)
        ds = np.round(self.model([[i[0], i[1]] for i in X] + [[i[1], i[0]] for i in X]), decimals=5)

        res = []
        for i in range(l):
            d = 0
            if X[i][0] != X[i][1]:
                d = np.round(self.doc2vec_weight * sc[i] + self.model_weight * (ds[i] + ds[l + i]), decimals=5)
                d = self.zero_point - np.log2(max(d, 10 ** -6))
            res.append(d)

        return res


class Dist3(Dist2):
    def __init__(self, model: Chainer, doc2vec: Doc2Vec):
        Dist2.__init__(self, model, doc2vec)
        self.doc2vec_weight = 1
        self.model_weight = 0

    def _scalar(self, x, y):
        v1 = self._embed(x)
        v2 = self._embed(y)
        return np.linalg.norm(v1 - v2)

    def dist(self, x: str, y: str) -> float:
        stat.dist_calls_count += 1

        if x == y:
            return 0

        sc = np.round(self._scalar(x, y), decimals=5)
        #ds = np.round(self.model([[x, y], [y, x]]), decimals=5)
        #d = np.round(ds[0] + ds[1], decimals=5)

        return self.doc2vec_weight * sc# + self.model_weight * (1 - np.log2(max(d, 10 ** -6)))

    def batch_dist(self, X: List[Tuple[str, str]]) -> List[float]:
        stat.dist_calls_count += len(X)

        l = len(X)
        sc = np.round([self._scalar(X[i][0], X[i][1]) for i in range(l)], decimals=5)
        ds = np.round(self.model([[i[0], i[1]] for i in X] + [[i[1], i[0]] for i in X]), decimals=5)

        res = []
        for i in range(l):
            d = 0
            if X[i][0] != X[i][1]:
                d = np.round(ds[i] + ds[l + i], decimals=5)
                d = self.doc2vec_weight * sc + self.model_weight * (1 - np.log2(max(d, 10 ** -6)))
            res.append(d)

        return res


class Dist4(Dist3):
    def __init__(self, model: Chainer, doc2vec: Doc2Vec):
        Dist3.__init__(self, model, doc2vec)
        self.doc2vec_weight = 1
        self.model_weight = 0

    def _scalar(self, x, y):
        v1 = self._embed(x)
        v2 = self._embed(y)
        return np.sqrt(np.dot(v1, v2) ** 2 / (np.dot(v1, v1) * np.dot(v2, v2)))


class Dist5(Dist2):
    def __init__(self, model: Chainer, doc2vec: Doc2Vec):
        Dist2.__init__(self, model, doc2vec)
        self.doc2vec_weight = 0.1
        self.model_weight = 1
        self.scale_edge = 0.95
        self.scale_coef = 0.01
        self.cache = LRU(1000000)

    def _scalar(self, x, y):
        v1 = self._embed(x)
        v2 = self._embed(y)
        return np.sqrt(np.dot(v1, v2) ** 2 / (np.dot(v1, v1) * np.dot(v2, v2)))

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


class Dist6(Dist5):
    def __init__(self, model: Chainer, doc2vec: Doc2Vec):
        Dist5.__init__(self, model, doc2vec)
        self.scale_edge = 0.88
        self.scale_coef = 0.12
        self.doc2vec_weight = 0.4
        self.model_weight = 1

    def _scalar(self, x, y):
        v1 = self._embed(x)
        v2 = self._embed(y)
        return np.arccos(np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))) / np.pi
