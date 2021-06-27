# CoNLL2003命名实体识别项目报告

## 前言

NER全称是命名实体识别（Named Entity Recognition, NER），目标是识别文本中感兴趣的实体，如位置、组织和时间。NER 是属于自然语言处理中的序列标注任务(sequence tagging)，序列标注中除了NER，还有如词性（POS）标记和分块（Chunking）等。已识别的实体可以在各种下游应用程序中使用，比如根据患者记录去识别和信息提取系统，也可以作为机器学习系统的特性，用于其他自然语言处理任务。NER总结的看其实就是提取出属于预定义类别的文本片段，它可能是通用性的，也可能是用户定义好的类型，属于特定的领域。

### 项目介绍

项目中的数据集使用的是CoNLL2003英文数据集，数据集包含训练集（14041个样本）、验证集（3250个样本）和测试集（3453个样本）。数据集中的目标实体分为人名（PER）、地名（LOC）、机构名（ORG），其他实体（MISC）。它使用标准的BIOUL实体标注方式，因此标注中包含 (B/L/U/I)-(PER/LOC/ORG/MISC) 这十六种标注以及 O 表示其他，共十七种标注类型。

项目的模型任务便是准确地识别出每一个实体以及它的类型，训练任务会在训练集和验证集上进行，通过验证集来优化模型参数，最后会在测试集上进行测试，计算Accuracy、Precision、Recall和F1等指标来评估模型。

通过调研NER的发展历程，我了解到NER的发展经历了从基于规则的线性模型，到后来的监督学习方法（HMM，DT，CRF等），再到近年来的深度学习方法的大流行。为了对比研究各类NER方法在CoNLL2003数据集上的效果，以及研究不同模型之间的异同和优劣，这里我们分别实现了传统的监督学习方法HMM和CRF来进行命名实体的识别，同时使用卷积神经网络、循环神经网络，以及预训练模型RoBERTa方法分别进行实体的识别。

在下一个部分的研究进展中，本文对NER方面的研究发展过程进行了简单的综述，方便理解命名实体识别领域的研究趋势；在实验模型介绍部分，我们针对各个模型的原理进行了简要的介绍和总结对比，希望能从模型原理上解释实验的结果差异；在实验结果部分，对各个模型的结果进行了总结和分析；最后，本文对命名实体识别项目做了一个简要的总结。

## 研究进展

命名实体识别NER的任务目标是给出一个命名实体的起始和终止边界，并给出命名实体的类别。一般而言完成NER任务的方法分为基于规则、基于无监督方法、基于特征的机器学习方法和基于深度学习的方法四种。其中一般领域性比较强，数据量很少的NER任务会用规则，其余基本上都是机器学习或者深度学习。尤其是在数据量比较充足的时候，深度学习一般都可以获得比较不错的效果。

在基于规则的NER任务中，需要手工指定符合条件的词及其对应的类别。具体使用的规则包括特定领域词典、同义词典、句法词汇模板和正则表达式等等。其优点在于不需要进行数据标注，但是指定规则工作量大，需要不断维护，同时迁移成本较高，常用的NER系统包括LaSIE-II, NetOwl等。当词汇表足够大时，基于规则的方法能够取得不错效果。但总结规则模板花费大量时间，且词汇表规模小，且实体识别结果普遍高精度、低召回。基于无监督的NER学习方法中使用聚类的方法，根据文本相似度进行不同实体类别组的聚类，同样不需要标注数据，但得到的结果准确度有限。常用到的特征或者辅助信息有词汇资源、语料统计信息（TF-IDF）、浅层语义信息（分块NP-chunking）等。基于特征的有监督学习方法中，NER任务可以视为机器学习token 级别的多分类任务或序列标注任务， 需要标注数据，同时一般结合精心设计的特征，包括词级别特征、文档特征和语料特征等等。常用的NER机器学习模型包括隐马尔可夫模型 HMM、决策树 DT、最大熵模型 MEM、最大熵马尔科夫模型 HEMM、支持向量机 SVM、条件随机场 CRF等。

深度学习NER受益于DL非线性，相比于传统线性模型可以学到更为复杂并对模型有益的特征，端到端过程得以实现，近年来成为主流研究方向。近年来，使用大规模的语料数据进行模型的预训练，然后在下游任务进行任务和数据导向的微调，成为了自然语言处理领域各个任务的主流方法。得益于大规模的语料，以及自注意力机制的存在，预训练模型能够很好地学习语言本身具有的含义，并且通过无监督的训练方式得到一个具备潜在语义信息的编码，这种通用的包含语义的编码能够适应大部分下游任务，也具有更强的表示能力。最具代表性的是BERT，它的提出是从监督学习方法到无监督预训练模型的转折。随后各类方法大多都是在Transformer基础上，调整预训练任务和策略等。比较有名的有GPT系列，XLNET，ALBERT，RoBERTa等，它们的核心模块都是自注意力机制，通用特点是使用大规模语料训练大规模的语言模型，提取通用的语言特征。目前大多数方法都是基于深度学习模型训练得到文本的隐语义编码，然后通过结合CRF模型得到预测结果。这样充分结合了深度学习强大的编码能力和传统机器学习的分类能力，也避免了深度学习模型在预测命名实体类别时，出现不可控的情况，例如B-PER后面不可能出现B-PER，这是一种规则约束，而单纯的深度学习模型是有可能出现这种情况的，结合CRF，通过学习得到一系列规则约束，避免了这些问题。

除了模型方面，近年来的研究偏向于结合不同级别的语义编码，从字符级别，到词级别，到实体级别的编码，不同模型之间的区别可能就在于如何提取这些编码，以及如何后处理这些编码。LUKE模型便是在其基础上对实体级别的编码进行预训练，并且使用BERT语言模型的掩码机制和新的训练任务，结合实体级别的自注意力机制，最终在多个实体相关的任务上获得了SOTA效果。

目前在命名实体识别领域，实体识别效果最好的是使用ACE模型(Automated Concatenation of Embeddings)结合文档级上下文语义，模型的目标是找到更好的embedding拼接方式，并且使用神经网络结构搜索的方式来自动化寻找拼接方式这一过程，而不是人为定义好拼接哪些embeddding。同时论文中使用了强化学习方法对模型进行训练，对于好的拼接方式，模型将会得到一个奖励，反之会得到一个惩罚，这样一个强化学习的奖励机制能够让模型学习到具体应该拼接哪些embedding。

## 实验模型介绍

模型（四个baseline）介绍框架embedding->encoder->tagger（系统主要模块流程），介绍baseline，再介绍自己的核心想法+伪代码

### HMM

### CRF

**一、框架**

1、data

数据读取预处理部分，主要包括tokens表示word level的处理，具体为对词进行小写处理，token_characters表示character-level的处理，type是dataset_reader的读取类型，设置为conll2003。

2、embedding

模型的第一层是词嵌入层，利用随机初始化的embedding矩阵将句子中的每个字由one-hot向量映射为低维稠密的字向量，其中每个维度都表示隐含的特征维度。单词的字符级表示与预训练得到的词向量连在一起作为最终的词表示。数据预处理采用word level进行，label的编码格式设置为BIOUL。

3、Tag decoder

模型的第二层是CRF层，进行句子级的序列标注。CRF层的参数是标签之间转移得分矩阵，进而在为一个位置进行标注的时候可以利用此前已经标注过的标签。条件随机场CRF利用全局信息进行标记用于解码。在预测当前标签时使用邻居的标签信息NER中，CRF模型关注整个句子的结构，是一个输出和输出直接相连的无向图，产生更高精度的标签。

4、trainer

训练器相关的参数的设置，使用SGD（随机梯度下降法）以0.015的学习率优化参数进行训练。

**二、baseline介绍**

条件随机场CRF是一种基于统计的序列标记和分割数据的方法，是用于序列标注问题的无向图模型，在给定需要标记的观测序列条件下，计算序列的联合概率。条件随机场的建立过程中，首先定义一个特征函数集，每个特征函数都以标注序列作为输入，提取特征作为输出.

条件随机场使用对数线性模型来计算给定观测序列下状态序列的条件概率$p(s|x;w)$。w是条件随机场模型的参数，可以视为每个特征函数的权重。CRF模型的训练其实就是对参数 w 的估计。模型训练结束之后，对给定的观测序列 x ，可得到其最优状态序列，解码后得到最终结果。



<img src="C:\Users\YMX\AppData\Roaming\Typora\typora-user-images\image-20210627092728258.png" alt="image-20210627092728258" style="zoom:67%;" />



在预测当前标签时使用邻居的标签信息NER中，CRF模型关注整个句子的结构，是一个输出和输出直接相连的无向图，产生更高精度的标签。同时，使用条件随机场CRF可解决tagging之间不独立的问题。对每种生成的tag序列，我们采用打分的方式代表该序列的好坏，分数越高代表当前生成的tag序列表现效果越好。

**三、伪代码**

![image-20210627134236115](C:\Users\YMX\AppData\Roaming\Typora\typora-user-images\image-20210627134236115.png)



### BiLSTM+CRF

### CNN+BiLSTM+CRF

**一、框架**

1、data

数据读取预处理部分，主要包括word level的处理和character-level的处理，具体处理同上。

2、embedding

模型的第一层是词嵌入层，利用预训练的embedding矩阵将句子中的每个字 由one-hot向量映射为低维稠密的字向量。输入的分布式通过把词映射到低维空间的稠密实值向量，其中每个维度都表示隐含的特征维度。词级别tokens采用glove embedding，字符级别采用multi-layer CNN 随机初始化，选择三元文法。通过卷积CNN得到的单词的字符级表示与预训练得到的词向量连在一起作为最终的词表示。

3、encoder

模型的第二层是双向LSTM层，自动提取句子特征。将一个句子的各个字的char embedding序列 (x1,x2,…,xn)作为双向LSTM各个时间步的输入，再将正向LSTM输出的隐状态序列 (h1⟶,h2⟶,…,hn⟶)与反向LSTM的 (h1⟵,h2⟵,…,hn⟵)在各个位置输出的隐状态进行按位置拼接得到完整的隐状态序列；在设置dropout后，接入一个线性层，将隐状态向量从m维映射到k维，k是标注集的标签数，从而得到自动提取的句子特征，记作矩阵P。可以把 pi的每一维pij都视作将字xi分类到第j个标签的打分值，如果再对 P进行Softmax的话，就相当于对各个位置独立进行k类分类。但是这样对各个位置进行标注时无法利用已经标注过的信息，所以接下来将接入一个CRF层来进行标注。

4、Tag decoder

模型的第三层是CRF层，进行句子级的序列标注。条件随机场CRF利用全局信息进行标记用于解码。在预测当前标签时使用邻居的标签信息NER中，CRF模型关注整个句子的结构，是一个输出和输出直接相连的无向图，产生更高精度的标签。

5、trainer

训练器相关的参数的设置。具体训练时使用SGD（随机梯度下降法）以0.015的学习率优化参数。 LSTM-CRF模型用前向和后向LSTM各一个独立层，维度为330，并加入了剔除率为0.5的dropout。validation_metric使用验证集矩阵计算F1精确度，evaluate_on_test在测试集上进行评估。

**二、baseline介绍**

CNN+BiLSTM+CRF用整个句子的信息来对词进行标记，其网络结构如下图所示，句子经过 embedding 层，一个 word 被表示为 N 维度的向量，随后利用 CNN 提取单词的字符级表示，字符级表示与 word 级表示连在一起作为最终的词表示。卷积层的输出大小与输入的句子长度有关，为了获取固定维度的句子表示，使用最大池化操作得到整个句子的全局特征。最后 tag decoder 使用该句子表示来得到标签的概率分布。

<img src="https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9WQmNEMDJqRmhnbWlhYUFrdDB5dVF5VVNiZFlEakFSQWliTHFpYjNlNDNZOGlheE9ZTFM4aWJ3eVBXWkRWZE9LRHRDZ09tWFppY2ZjQW9GNURqTnZvYkk1TXdCZy82NDA?x-oss-process=image/format,png" alt="img" style="zoom:50%;" />

**三、伪代码**

![image-20210627134555427](C:\Users\YMX\AppData\Roaming\Typora\typora-user-images\image-20210627134555427.png)

### BERT+BiLSTM+CRF

**一、框架**

1、data

数据读取预处理部分，主要包括 tokens 和 token_characters，进行词级别的处理和字符级别的处理，另外加入预训练模型transformer中的bert-base-cased。

2、embedding

模型的第一层是词嵌入层，利用预训练的embedding矩阵将句子中的每个字 由one-hot向量映射为低维稠密的字向量。输入的分布式通过把词映射到低维空间的稠密实值向量，其中每个维度都表示隐含的特征维度。词级别tokens采用预训练模型bert（BERTbase: L=12, H=768, A=12, Total Parameters=110M）进行embedding处理。通过LSTM得到的单词的字符级表示与预训练得到的词向量连在一起作为最终的词表示。

3、encoder

模型的第二层是双向LSTM层，自动提取句子特征。具体过程同CNN+BiLSTM+CRF。

4、Tag decoder

模型的第三层是CRF层，进行句子级的序列标注。具体过程同CNN+BiLSTM+CRF。

5、trainer

训练器相关的参数的设置。具体训练时使用ADAM以5e-07的学习率优化参数，以5.0作为梯度的阈值。validation_metric使用验证集矩阵计算F1精确度，evaluate_on_test在测试集上进行评估。

**二、baseline介绍**

BERT+BiLSTM+CRF中，BERT负责学习输入句子中每个字和符号到对应的实体标签的规律，而CRF负责学习相邻实体标签之间的转移规则。

![image-20210627103717625](C:\Users\YMX\AppData\Roaming\Typora\typora-user-images\image-20210627103717625.png)

BERT预训练模型的数据集包括200k训练单词，其中标注为五类： Person, Organization, Location,Miscellaneous, or Other (non-named entity)，在微调的时候，在BERT上加了一个分类层来判断是否是名字的一部分，如下图所示：

![img](https://img-blog.csdnimg.cn/20190523180916640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc0MDA4Mg==,size_16,color_FFFFFF,t_70)

BERT层学到了句子中每个字符最可能对应的实体标注是什么，这个过程是考虑到了每个字符左边和右边的上下文信息的，通过引入CRF解决输出的最大分数对应的实体标注依然可能有误的问题。由BERT学习序列的状态特征，从而得到一个状态分数，该分数直接输入到CRF层，省去了人工设置状态特征模板。Bert+CRF中，状态分数是根据训练得到的BERT模型的输出计算出来的，转移分数是从CRF层提供的转移分数矩阵得到的。

**三、伪代码**

![image-20210627134608034](C:\Users\YMX\AppData\Roaming\Typora\typora-user-images\image-20210627134608034.png)

### RoBERTa+BiLSTM+CRF

## 实验结果分析

### 数据集

命名实体识别NER任务中，我们使用来自 CoNLL 2003 共享的的英文数据进行实验，该数据集包含四种不同类型的命名实体：PERSON、LOCATION、ORGANIZATION 和 MISC。 我们使用 BIOUL编码格式，因为之前的研究中相对于默认的BIO编码格式存在显著改进。

### 实验

我们将所选的六个模型的性能进行比较——HMM，CRF，BiLSTM，BiLSTM+CRF，CNN+BiLSTM+CRF，BERT+BiLSTM+CRF，RoBERTa+BiLSTM+CRF对字符级信息进行建模。除传统机器学习模型之外，基于所有这些模型都使用斯坦福大学的 GloVe 词嵌入和相同的超参数运行，如表 1 所示。 

我们的模型可以通过 GloVe 嵌入获得 91.22 的最佳 F1 分数。使用了各种机器学习分类器的组合，根据表1结果，BLSTM-CRF模型明显优于 CRF 模型，表明句子特征提取对于命名实体识别任务很重要。BiLSTM+CRF 略微优于 CNN+BiLSTM+CRF，可能是使用了GloVe 不同的词嵌入。然而，BERT+BiLSTM+CRF 明显优于 BiLSTM+CRF，预训练模型BERT的引入大大提高了准确度，而针对BERT模型的改进模型RoBERTa性能也有所提升。

![image-20210627153918406](C:\Users\YMX\AppData\Roaming\Typora\typora-user-images\image-20210627153918406.png)

## 总结

## 参考文献

[1] A survey of named entity recognition and classification

[2] A survey on deep learning for named entity recognition

[3] Neural architectures for named entity recognition

[4] Named entity recognition with bidirectional lstm-cnns

[5] Bidirectional lstm-crf models for sequence tagging

[6] End-to-end sequence labeling via bidirectional lstm-cnns-crf

[7] Semi-supervised multitask learning for sequence labeling

[8] Bert: Pretraining of deep bidirectional transformers for language understanding