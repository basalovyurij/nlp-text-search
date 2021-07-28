import deeppavlov
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from nlp_text_search import create_settings, LinearizedDist, DefaultSearchEngine
from nlp_text_search.vptree.jvptree import RandomVantagePointSelectionStrategy
from unittest import TestCase


class TestSearchEngine(TestCase):
    def __init__(self):
        TestCase.__init__(self, 'test')

    def test(self):
        se = self._create_se()
        res = self._round(se.search('красная ручка', 5))
        print(res)

        se.save('./se', copy_model=True)
        se2 = DefaultSearchEngine.load('./se')
        res2 = self._round(se2.search('красная ручка', 5))
        print(res2)

        self.assertListEqual(res, res2)

    def _round(self, res):
        return list([(t[0], round(t[1], 4)) for t in res])

    def _create_se(self) -> DefaultSearchEngine:
        paraphrases, all_texts = self._get_data()
        settings = create_settings(paraphrases, 'test', root_path='d:\\deeppavlov_data')
        deeppavlov.train_model(settings, download=True)
        doc2vec = Doc2Vec([TaggedDocument(simple_preprocess(t), [i]) for i, t in enumerate(all_texts)],
                          min_count=1, workers=1, negative=0, dm=0, hs=1)
        return DefaultSearchEngine(settings, doc2vec, LinearizedDist, points=all_texts,
                                   vantage_point_selection_strategy=RandomVantagePointSelectionStrategy())

    def _get_data(self):
        paraphrases = [
            (('красная ручка', 'синяя ручка'), 1),
            (('красная ручка', 'зеленая ручка'), 1),
            (('красная машина', 'синяя машина'), 1),
            (('красная машина', 'зеленая машина'), 1),
            (('синяя ручка', 'красная ручка'), 1),
            (('синяя ручка', 'зеленая ручка'), 1),
            (('синяя машина', 'красная машина'), 1),
            (('синяя машина', 'зеленая машина'), 1),
            (('красная ручка', 'красная машина'), 0),
            (('красная ручка', 'синяя машина'), 0),
            (('красная ручка', 'зеленая машина'), 0),
            (('синяя ручка', 'красная машина'), 0),
            (('синяя ручка', 'синяя машина'), 0),
            (('синяя ручка', 'зеленая машина'), 0),
            (('зеленая ручка', 'красная машина'), 0),
            (('зеленая ручка', 'синяя машина'), 0),
            (('зеленая ручка', 'зеленая машина'), 0)
        ]
        all_texts = list(set([t[0][0] for t in paraphrases] + [t[0][1] for t in paraphrases]))
        return paraphrases, all_texts
