{
    "dataset_reader": {
        "type": "conll2003",
        "coding_scheme": "BIOUL",
        "tag_label": "ner",
        "token_indexers": {
            "token_characters": {
                "type": "characters",
                "min_padding_length": 4
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
            "type": "pass_through",
            "input_dim": 100
        },
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "text_field_embedder": {
            "token_embedders": {
                "token_characters": {
                    "type": "empty"
                },
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 100
                }
            }
        }
    },
    "train_data_path": "../data/eng.train",
    "validation_data_path": "../data/eng.testa",
    "test_data_path": "../data/eng.testb",
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 100,
        "optimizer": {
            "type": "sgd",
            "lr": 0.015
        }
    },
    "data_loader": {
        "batch_size": 32,
        "shuffle": true
    },
    "datasets_for_vocab_creation": [
        "train"
    ],
    "evaluate_on_test": true
}