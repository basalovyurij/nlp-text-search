from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from nlp_text_search import create_settings, LinearizedDist, SaveableVPTreeSearchEngine

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
doc2vec = Doc2Vec([TaggedDocument(simple_preprocess(t), [i]) for i, t in enumerate(all_texts)], min_count=1)

se = SaveableVPTreeSearchEngine(settings, doc2vec, LinearizedDist)
se.fit(all_texts)
print(se.search('красная ручка', 3))

# se.save('./se')
# se = SaveableVPTreeSearchEngine.load('./se')

#se = SaveableVPTreeSearchEngine(deeppavlov.configs.ranking.paraphrase_ident_paraphraser_pretrain, doc2vec, Dist6)
#se.fit([i['name'] for i in train])