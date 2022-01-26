# PSPE
Source code of EMNLP2020 paper ["Pre-training Entity Relation Encoder with Intra-span and Inter-spanInformation"](https://aclanthology.org/2020.emnlp-main.132/).

It is based on our [NERE](https://github.com/Receiling/NERE) toolkit.

## Requirements
* `python`: 3.7.6
* `pytorch`: 1.4.0
* `transformers`: 2.8.0
* `configargparse`: 1.1
* `bidict`: 0.18.0
* `fire`: 0.2.1

## Pre-training
Before pre-training, please prepare a pre-training corpus (e.g. Wikipedia), the format of the pre-training corpus must be the same as the file [`data/wiki/wikipedia_sentences.txt`](https://github.com/Receiling/PSPE/blob/master/data/wiki/wikipedia_sentences.txt).

Then preprocess the pre-training corpus for convenience:
```bash
$ python inputs/preprocess.py contrastive_loss_preprocess \
                            data/wiki/wikipedia_sentences.txt \
                            data/wiki/wikipedia_pretrain.json \
                            data/bert_base_cased_vocab.json
```

Pre-training:
```bash
$ PYTHONPATH=$(pwd) python examples/entity_relation_pretrain_nce/entity_relation_extractor_pretrain_nce.py \
                            --config_file examples/entity_relation_pretrain_nce/config.yml \
                            --device 0 \
                            --fine_tune
```

## Fine-tuning
```bash
$ mkdir pretrained_models
$ cd pretrained_models
```
Before fine-tuning, please download the pre-trained model [`SPE`](https://pan.baidu.com/s/1kWZqaknh-Lg4d5XCGHoXOQ)(password: dct8), and place the pre-trained model in the folder `pretrained_models`. And make sure that the format of the dataset must be the same as [`data/demo/train.json`](https://github.com/Receiling/PSPE/blob/master/data/demo/train.json).
```bash 
PYTHONPATH=$(pwd) python examples/attention_entity_relation/att_entity_relation_extractor.py \
                        --config_file examples/attention_entity_relation/config.yml \
                        --device 0 \
                        --fine_tune
```

## Cite
If you find our code is useful, please cite:
```
@inproceedings{wang-etal-2020-pre,
    title = "Pre-training Entity Relation Encoder with Intra-span and Inter-span Information",
    author = "Wang, Yijun  and
      Sun, Changzhi  and
      Wu, Yuanbin  and
      Yan, Junchi  and
      Gao, Peng  and
      Xie, Guotong",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.132",
    doi = "10.18653/v1/2020.emnlp-main.132",
    pages = "1692--1705",
    abstract = "In this paper, we integrate span-related information into pre-trained encoder for entity relation extraction task. Instead of using general-purpose sentence encoder (e.g., existing universal pre-trained models), we introduce a span encoder and a span pair encoder to the pre-training network, which makes it easier to import intra-span and inter-span information into the pre-trained model. To learn the encoders, we devise three customized pre-training objectives from different perspectives, which target on tokens, spans, and span pairs. In particular, a span encoder is trained to recover a random shuffling of tokens in a span, and a span pair encoder is trained to predict positive pairs that are from the same sentences and negative pairs that are from different sentences using contrastive loss. Experimental results show that the proposed pre-training method outperforms distantly supervised pre-training, and achieves promising performance on two entity relation extraction benchmark datasets (ACE05, SciERC).",
}
```



