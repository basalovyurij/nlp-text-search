ROOT_PATH = "/media/yurij/WORK/Projects/STAR/deeppavlov_data"


def create_settings(tag, name, data_size=100000, max_sequence_length=20, nn_class_name='mpm_nn', dataset_path=None,
                    hidden_dim=200, aggregation_dim=200, embedder='fasttext', embedding_path=None):
    if not dataset_path:
        dataset_path = name + "/dataset.json"

    preproc = {
        "id": "preproc",
        "class_name": "siamese_preprocessor",
        "use_matrix": False,
        "max_sequence_length": max_sequence_length,
        "fit_on": ["x"],
        "in": ["x"],
        "out": ["x_proc"],
        "sent_vocab": {
            "id": "siam_sent_vocab",
            "class_name": "simple_vocab",
            "save_path": "{MODELS_PATH}/punch_outing/" + name + "/sent.dict",
            "load_path": "{MODELS_PATH}/punch_outing/" + name + "/sent.dict"
        },
        "tokenizer": {
            "class_name": "nltk_tokenizer"
        },
        "vocab": {
            "id": "siam_vocab",
            "class_name": "simple_vocab",
            "save_path": "{MODELS_PATH}/punch_outing/" + name + "/tok.dict",
            "load_path": "{MODELS_PATH}/punch_outing/" + name + "/tok.dict"
        }
    }

    embedding = {
        "id": "embeddings",
        "class_name": "emb_mat_assembler",
        "embedder": "#siam_embedder",
        "vocab": "#siam_vocab"
    }

    nn = {
        "id": "model",
        "in": ["x_proc"],
        "in_y": ["y"],
        "out": ["y_predicted"],
        "class_name": nn_class_name,
        "len_vocab": "#siam_vocab.len",
        "use_matrix": False,
        "attention": True,
        "emb_matrix": "#embeddings.emb_mat",
        "embedding_dim": "#siam_embedder.dim",
        "aggregation_dim": aggregation_dim,
        "max_sequence_length": "#preproc.max_sequence_length",
        "seed": 243,
        "hidden_dim": hidden_dim,
        "learning_rate": 1e-3,
        "triplet_loss": False,
        "batch_size": 256,
        "save_path": "{MODELS_PATH}/punch_outing/" + name + "/model_weights.h5",
        "load_path": "{MODELS_PATH}/punch_outing/" + name + "/model_weights.h5",
        "preprocess": "#preproc.__call__"
    }

    if embedder == 'fasttext':
        if not embedding_path:
            embedding_path = "{DOWNLOADS_PATH}/embeddings/ft_native_300_ru_wiki_lenta_lower_case.bin"

        preproc["embedder"] = {
            "id": "siam_embedder",
            "class_name": "fasttext",
            "load_path": embedding_path
        }
    if embedder == 'word2vec':
        preproc["embedder"] = {
            "id": "siam_embedder",
            "class_name": "word2vec_embedder",
            "load_path": embedding_path
        }

    pipe = [preproc, embedding, nn]

    return {
        "dataset_reader": {
            "class_name": "mssql_reader",
            "tag": tag,
            "size": data_size,
            "data_path": "{MODELS_PATH}/punch_outing/" + dataset_path
        },
        "dataset_iterator": {
            "class_name": "siamese_iterator",
            "seed": 243
        },
        "chainer": {
            "in": ["x"],
            "in_y": ["y"],
            "pipe": pipe,
            "out": ["y_predicted"]
        },
        "train": {
            "epochs": 10,
            "batch_size": 256,
            "pytest_max_batches": 2,
            "train_metrics": ["f1", "acc", "log_loss"],
            "metrics": ["f1", "acc", "log_loss"],
            "validation_patience": 10,
            "val_every_n_epochs": 1,
            "log_every_n_batches": 24,
            "class_name": "nn_trainer",
            "evaluation_targets": [
                "test"
            ]
        },
        "metadata": {
            "variables": {
                "ROOT_PATH": ROOT_PATH,
                "DOWNLOADS_PATH": "{ROOT_PATH}/downloads",
                "MODELS_PATH": "{ROOT_PATH}/models"
            },
            "requirements": [
                "{DEEPPAVLOV_PATH}/requirements/tf.txt",
                "{DEEPPAVLOV_PATH}/requirements/fasttext.txt"
            ],
            "download": []
        }
    }