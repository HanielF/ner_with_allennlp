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
  TransformerCoNLL2003DatasetReader(model_name, max_length)::{
    type: 'conll2003',
    tag_label: 'ner',
    coding_scheme: 'BIOUL',
    token_indexers: {
      tokens: {
        // 输入的是words但是Transformer需要wordpieces，有不一致的情况
        // pretrained_transformer_mismatched indexer可以将单个word分成wordpieces
        // embedding之后拼在一起组成single word embedding
        type: 'pretrained_transformer_mismatched',
        model_name: model_name,
        max_length: max_length,
      },
    },
  }
}
