# Fulltext-like search using NLP concept

![Python 3.6, 3.7](https://img.shields.io/badge/python-3.6%20%7C%203.7-green.svg)

Library for fulltext search using NLP concept. Use [deeppavlov](https://deeppavlov.ai)
for paraphrase identification and Vantage-Point tree (based on [jvptree](https://github.com/jchambers/jvptree/)) for fast search.

## Installation

Install and update using pip:
```
pip install -U nlp-text-search
```

## Usage

First init data, create [deeppavlov](https://deeppavlov.ai) settings and 
Doc2Vec for emdedding.
```
import deeppavlov
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
deeppavlov.train_model(settings)
doc2vec = Doc2Vec([TaggedDocument(simple_preprocess(t), [i]) for i, t in enumerate(all_texts)],
                  min_count=1, workers=1, negative=0, dm=0, hs=1)
```

Then create search engine and search nearest neighbors
```
se = DefaultSearchEngine(settings, doc2vec, LinearizedDist, points=all_texts)
print(se.search('красная ручка', 4))
```
returns
```
[('красная ручка', 0), ('зеленая ручка', 0.05778998136520386), ('синяя ручка', 0.06721997261047363), ('синяя машина', 0.48162001371383667)]
```

You also can save and load search engine
```
se.save('se')
se = DefaultSearchEngine.load('se')
```