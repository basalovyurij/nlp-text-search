from abc import ABCMeta, abstractmethod
from deeppavlov.core.common.chainer import Chainer
from gensim.models import Doc2Vec
from itertools import groupby
from lru import LRU
from methodtools import lru_cache
import nltk
import numpy as np
from typing import Any, List, Tuple


class Stat:
    def __init__(self):
        self.dist_calls_count = 0


stat = Stat()


class Dist(metaclass=ABCMeta):
    @abstractmethod
    def dist(self, x: Any, y: Any) -> float:
        pass

    @abstractmethod
    def batch_dist(self, x: List[Tuple[Any, Any]]) -> List[float]:
        pass


class LinearizedDist(Dist):
    def __init__(self, model: Chainer, doc2vec: Doc2Vec, linearization_settings: dict):
        self.model = model
        self.doc2vec = doc2vec

        self.scale_steps = []
        for k, v in groupby(self._get_scales(linearization_settings), key=lambda x: x[0]):
            steps = list(set(v))
            if len(steps) > 1:
                raise ValueError('Ambigious scales: %s' % ', '.join(['%s -> %s' % (x[0], x[1]) for x in steps]))
            self.scale_steps.append(steps[0])

        self.scale_lin_coefs = []
        for i in range(len(self.scale_steps) - 1):
            fr = self.scale_steps[i + 1][0] - self.scale_steps[i][0]
            to = self.scale_steps[i + 1][1] - self.scale_steps[i][1]
            self.scale_lin_coefs.append(to / fr)

        self.doc2vec_weight = linearization_settings.get('doc2vec_weight', 0.1)
        self.model_weight = linearization_settings.get('model_weight', 1)

        self.cache = LRU(1000000)

    def _get_scales(self, linearization_settings: dict):
        start_scale = (0.0, 0.0)
        end_scale = (1.0, 1.0)
        scale_steps: List[Tuple[float, float]] = [start_scale]

        if 'scale_edge' in linearization_settings or 'scale_coef' in linearization_settings:
            if 'scales_from' in linearization_settings or 'scales_to' in linearization_settings:
                raise ValueError('Specify either scale_edge/scale_coef or scales_from/scales_to')

            scale_steps.append((linearization_settings.get('scale_edge', 1.0), linearization_settings.get('scale_coef', 1.0)))

        elif 'scales_from' in linearization_settings or 'scales_to' in linearization_settings:
            if 'scale_edge' in linearization_settings or 'scale_coef' in linearization_settings:
                raise ValueError('Specify either scale_edge/scale_coef or scales_from/scales_to')

            scales_from = linearization_settings.get('scales_from', [])
            scales_to = linearization_settings.get('scales_to', [])
            if not isinstance(scales_from, list) or not isinstance(scales_to, list):
                raise ValueError('Parameters scales_from/scales_to must be list of floats')
            if len(scales_from) != len(scales_to):
                raise ValueError('Length of scales_from/scales_to must be equal')

            for i in range(len(scales_from)):
                scale_steps.append((scales_from[i], scales_to[i]))

        scale_steps.append(end_scale)

        return scale_steps

    def get_linearization_settings(self):
        return {
            'scale_steps': self.scale_steps,
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
        # linearizing distribution
        for i in range(len(self.scale_steps) - 1):
            if self.scale_steps[i][0] <= d < self.scale_steps[i + 1][0]:
                return self.scale_steps[i][1] + self.scale_lin_coefs[i] * (d - self.scale_steps[i][0])
        return d

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

    def batch_dist(self, x: List[Tuple[str, str]]) -> List[float]:
        res = [0] * len(x)
        new_pairs = []
        y = []
        for i, (xi, yi) in enumerate(x):
            d = self._get_from_cache(xi, yi)
            if d is not None:
                res[i] = d
            else:
                new_pairs.append(i)
                y.append((xi, yi))

        sc = np.round([self._scalar(xi, yi) for (xi, yi) in y], decimals=5)
        ds = np.round(self.model([[xi, yi] for (xi, yi) in y] + [[yi, xi] for (xi, yi) in y]), decimals=5)

        l = len(new_pairs)
        for i, (xi, yi) in enumerate(y):
            d = self._combine(sc[i], ds[i], ds[l + i])
            self.cache[(xi, yi)] = d
            res[new_pairs[i]] = d

        return res
