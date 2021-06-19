local common = import 'common.libsonnet';

{
  dataset_reader : common.BaseCoNLL2003DatasetReader(),
  datasets_for_vocab_creation: ['train'],
  train_data_path: './data/eng.train',
  validation_data_path: './data/eng.testa',
  test_data_path: './data/eng.testb',
  evaluate_on_test: true,
  model: {
    type: 'crf_tagger',
    label_encoding: 'BIOUL',
    constrain_crf_decoding: true,
    calculate_span_f1: true,
    dropout: dropout,
    include_start_end_transitions: false,
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 100,
          pretrained_file: 'https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz',
          trainable: true
        }
        token_characters: {
          type: 'character_encoding',
          embedding: {
            embedding_dim: 25,
          },
          encoder: {
            type: 'lstm',
            input_size: '25',
            hidden_size: '25',
            bidirectional: true,
            num_layers: 1,
            dropout: 0.5
          }
        },
      },
    },
    encoder:{
      type: 'lstm',
      input_size: 25,
      hidden_size: 200,
      bidirectional: true,
      num_layers: 1,
      dropout: 0.5,
    }
  },
  data_loader:{
    batch_size: 32,
    shuffle: true
  },
  trainer: {
    optimizer:{
      type: sgd,
      lr: 0.015,
    },
    num_epochs: 150,
    cuda_device: 0
  }
}
