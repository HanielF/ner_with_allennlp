local common = import 'common.libsonnet';

// local model_name = 'julien-c/flair-ner';
// local model_name = 'studio-ousia/luke-base';
local model_name = 'roberta-large';
local max_length = 512;

local embedding_dim = 1024;
local dropout = 0.5;

local lstm_hidden_size = 512;
local lstm_dropout = 0.5;
local lstm_layer = 1;
local lstm_bidirectional = true;

local fordward_input_dim = lstm_hidden_size*2;
local fordward_layer = 2;
local forward_hidden_dim = [fordward_input_dim, 128];
local activation = 'relu';

local optimizer = 'adam';
local batch_size = 8;
local lr = 1e-5;
local num_epochs = 10;
local cuda_device = 0;

{
  dataset_reader : common.TransformerCoNLL2003DatasetReader(model_name, max_length),
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
          type: 'pretrained_transformer_mismatched',
          model_name: model_name,
          max_length: max_length,
          train_parameters: true
        },
      },
    },
    encoder:{
      type: 'lstm',
      input_size: embedding_dim,
      hidden_size: lstm_hidden_size,
      bidirectional: lstm_bidirectional,
      num_layers: lstm_layer,
      dropout: lstm_dropout,
    },
    feedforward:{
      "input_dim":fordward_input_dim,
      "num_layers":fordward_layer,
      "hidden_dims":forward_hidden_dim,
      "activations":activation,
      "dropout":dropout,
    },
    "regularizer":{
        "regexes": [["weight","l2"]]
    }
  },
  data_loader:{
    batch_size: batch_size,
    shuffle: true
  },
  trainer: {
    optimizer:{
      type: optimizer,
      lr: lr,
    },
    num_epochs: num_epochs,
    cuda_device: cuda_device
  },
  /* "distributed": { */
      /* "cuda_devices": [0, 1], */
  /* } */
}
