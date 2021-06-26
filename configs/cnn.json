{
    "dataset_reader": {
        "type": "conll2003",
        "coding_scheme": "BIOUL",
        "tag_label": "ner",
        "token_indexers": {
            "token_characters": {
                "type": "characters",
                "min_padding_length": 3
            },
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "model": {
        "type": "crf_tagger",
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "dropout": 0.5,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.5,
            "hidden_size": 200,
            "input_size": 330,
            "num_layers": 1
        },
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "text_field_embedder": {
            "token_embedders": {
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 30
                    },
                    "encoder": {
                        "type": "cnn",
                        "conv_layer_activation": "relu",
                        "embedding_dim": 30,
                        "ngram_filter_sizes": [
                            3
                        ],
                        "num_filters": 30
                    }
                },
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "./glove.6B.300d.txt",
                    "trainable": true
                }
            }
        }
    },
    "train_data_path": "../data/eng.train",
    "validation_data_path": "../data/eng.testa",
    "test_data_path": "../data/eng.testb",
    "trainer": {
        "checkpointer": {
            "num_serialized_models_to_keep": 3
        },
        "cuda_device": 0,
        "num_epochs": 150,
        "optimizer": {
            "type": "sgd",
            "lr": 0.015
        },
        "patience": 25,
        "validation_metric": "+f1-measure-overall"
    },
    "data_loader": {
        "batch_size": 10
    },
    "datasets_for_vocab_creation": [
        "train"
    ],
    "evaluate_on_test": true
}