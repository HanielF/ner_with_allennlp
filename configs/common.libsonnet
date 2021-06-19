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
        type: 'single_id',
        lowercase_tokens: true,
      },
      // allennlp-source-code/data/token_indexers/token_characters_indexer.py
      token_characters: {
        type: 'characters',
        min_padding_length: 3,
      },
    },
  },
}
