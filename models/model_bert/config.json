{
    "dataset_reader": {
        "type": "conll2003",
        "coding_scheme": "BIOUL",
        "tag_label": "ner",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "max_length": 512,
                "model_name": "bert-base-cased"
            }
        }
    },
    "model": {
        "type": "crf_tagger",
        "calculate_span_f1": true,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.0,
            "hidden_size": 576,
            "input_size": 768,
            "num_layers": 2
        },
        "label_encoding": "BIOUL",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "max_length": 512,
                    "model_name": "bert-base-cased"
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
        "num_epochs": 75,
        "optimizer": {
            "type": "adam",
            "lr": 5e-07
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