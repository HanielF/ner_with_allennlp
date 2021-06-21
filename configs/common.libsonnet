{
  // allennlp-source-code/data/dataset_readers/conll2003
  BaseCoNLL2003DatasetReader()::{
    type: 'conll2003',
    tag_label: 'ner',
    coding_scheme: 'BIOUL',
    token_indexers: {
      // indexer in tokens and token_characters namespaces
      tokens: {
        // allennlp-source-code/data/token_indexers/single_id_token_indexer.py
        // 默认用single_id tokenizer
        type: 'single_id',
        lowercase_tokens: true,
      },
      // allennlp-source-code/data/token_indexers/token_characters_indexer.py
      token_characters: {
        // 默认用CharacterTokenizer()
        type: 'characters',
        min_padding_length: 3,
      },
    },
  },


  BERTDatasetReader(bert_model, max_length)::{
    type: 'conll2003',
    tag_label: 'ner',
    coding_scheme: 'BIOUL',
    token_indexers: {
      tokens: {
        type: 'pretrained_transformer_mismatched',
        model_name: bert_model,
        max_length: max_length,
      },
    },
}
