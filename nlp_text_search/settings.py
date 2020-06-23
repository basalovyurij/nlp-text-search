from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
import json
import os
import random
from typing import Any, Dict, List, Tuple


@register('simple_reader')
class SimpleDataReader(DatasetReader):
    def save(self, data: List[Tuple[Tuple[str, str], int]], data_path: str, train_size: float):
        data_path = os.path.abspath(data_path)
        if not os.path.isdir(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))

        random.shuffle(data)

        l = int(len(data) * train_size)
        dataset = {
            'train': data[0:l],
            'valid': data[l:],
            'test': data[l:]
        }
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4, sort_keys=True)

    def read(self, data_path: str, *args, **kwargs) -> Dict[str, List[Tuple[Any, Any]]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
        
def create_settings(paraphrases: List[Tuple[Tuple[str, str], int]], name: str, train_size=0.8,
                    fasttext_embed_path=None, root_path='~/.deeppavlov',
                    max_sequence_length=30, nn_class_name='mpm_nn', hidden_dim=200, aggregation_dim=200):

    downloads = []
    if fasttext_embed_path is None:
        fasttext_embed_path = '{DOWNLOADS_PATH}/embeddings/ft_native_300_ru_wiki_lenta_lower_case.bin'
        downloads.append({
            'url': 'http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lower_case/ft_native_300_ru_wiki_lenta_lower_case.bin',
            'subdir': '{DOWNLOADS_PATH}/embeddings'
        })

    preproc = {
        'id': 'preproc',
        'class_name': 'siamese_preprocessor',
        'use_matrix': False,
        'max_sequence_length': max_sequence_length,
        'fit_on': ['x'],
        'in': ['x'],
        'out': ['x_proc'],
        'sent_vocab': {
            'id': 'siam_sent_vocab',
            'class_name': 'simple_vocab',
            'save_path': '{MODELS_PATH}/%s/sent.dict' % name,
            'load_path': '{MODELS_PATH}/%s/sent.dict' % name
        },
        'tokenizer': {
            'class_name': 'nltk_tokenizer'
        },
        'vocab': {
            'id': 'siam_vocab',
            'class_name': 'simple_vocab',
            'save_path': '{MODELS_PATH}/%s/tok.dict' % name,
            'load_path': '{MODELS_PATH}/%s/tok.dict' % name
        }
    }

    embedding = {
        'id': 'embeddings',
        'class_name': 'emb_mat_assembler',
        'embedder': '#siam_embedder',
        'vocab': '#siam_vocab'
    }

    nn = {
        'id': 'model',
        'in': ['x_proc'],
        'in_y': ['y'],
        'out': ['y_predicted'],
        'class_name': nn_class_name,
        'len_vocab': '#siam_vocab.len',
        'use_matrix': False,
        'attention': True,
        'emb_matrix': '#embeddings.emb_mat',
        'embedding_dim': '#siam_embedder.dim',
        'aggregation_dim': aggregation_dim,
        'max_sequence_length': '#preproc.max_sequence_length',
        'seed': 243,
        'hidden_dim': hidden_dim,
        'learning_rate': 1e-3,
        'triplet_loss': False,
        'batch_size': 256,
        'save_path': '{MODELS_PATH}/%s/model_weights.h5' % name,
        'load_path': '{MODELS_PATH}/%s/model_weights.h5' % name,
        'preprocess': '#preproc.__call__'
    }

    preproc['embedder'] = {
        'id': 'siam_embedder',
        'class_name': 'fasttext',
        'load_path': fasttext_embed_path
    }

    pipe = [preproc, embedding, nn]

    res = {
        'dataset_reader': {
            'class_name': 'simple_reader',
            'data_path': '{MODELS_PATH}/%s/dataset.json' % name
        },
        'dataset_iterator': {
            'class_name': 'siamese_iterator',
            'seed': 243
        },
        'chainer': {
            'in': ['x'],
            'in_y': ['y'],
            'pipe': pipe,
            'out': ['y_predicted']
        },
        'train': {
            'epochs': 10,
            'batch_size': 256,
            'pytest_max_batches': 2,
            'train_metrics': ['f1', 'acc', 'log_loss'],
            'metrics': ['f1', 'acc', 'log_loss'],
            'validation_patience': 10,
            'val_every_n_epochs': 1,
            'log_every_n_batches': 1,
            'class_name': 'nn_trainer',
            'evaluation_targets': [
                'test'
            ]
        },
        'metadata': {
            'variables': {
                'ROOT_PATH': root_path,
                'DOWNLOADS_PATH': '{ROOT_PATH}/downloads',
                'MODELS_PATH': '{ROOT_PATH}/models'
            },
            'requirements': [],
            'download': downloads
        }
    }

    SimpleDataReader().save(paraphrases, parse_config(res)['dataset_reader']['data_path'], train_size)

    return res