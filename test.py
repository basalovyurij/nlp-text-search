import deeppavlov
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from nlp_text_search import create_settings, LinearizedDist, DefaultSearchEngine

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
    (('синяя ручка', 'зеленая машина'), 0)
]
all_texts = list(set([t[0][0] for t in paraphrases] + [t[0][1] for t in paraphrases]))

settings = create_settings(paraphrases, 'test')
deeppavlov.train_model(settings)
doc2vec = Doc2Vec([TaggedDocument(simple_preprocess(t), [i]) for i, t in enumerate(all_texts)],
                  min_count=1, workers=1, negative=0, dm=0, hs=1)

se = DefaultSearchEngine(settings, doc2vec, LinearizedDist, points=all_texts)
print(se.search('красная ручка', 5))

se2 = DefaultSearchEngine.load('./se')
print(se2.search('красная ручка', 5))
