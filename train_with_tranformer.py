import torch
import nlpaug
import flair
from torch.optim.adam import Adam

from flair.data import Corpus, Token
from flair.datasets import CONLL_03, SentenceDataset
from typing import List

from mapping import (
        twitter_ner_mapped,
        onto_ner_mapped,
        wikigold_ner_mapped,
        webpages_ner_mapped
    )


flair.device = 'cuda:0'
from nlpaugment import(
    punctuation_aug,
    capitalization_aug,
    ocr_aug,
    keyboard_aug,
    random_insert_aug,
    random_subtitute_aug,
    random_swap_aug,
    random_delete_aug
)

from flair.embeddings import(
        TransformerDocumentEmbeddings,
        TokenEmbeddings,
        WordEmbeddings,
        StackedEmbeddings,
        FlairEmbeddings,
        CharacterEmbeddings
)

from flair.training_utils import EvaluationMetric
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


# 1. get the corpus

dataset_name = "conll3"


for seed in [1,2,3]:
    flair.set_seed(123)

    if dataset_name == "onto_ner":

        corpus = onto_ner_mapped()
    elif dataset_name == "conll3":
        corpus = CONLL_03()
    elif dataset_name == "wikipedia":
        corpus = wikigold_ner_mapped()
    elif dataset_name == "webpages":
        corpus = webpages_ner_mapped()
    elif dataset_name == "twitter":
        corpus = twitter_ner_mapped()

    flair.set_seed(seed)

    # 2. what tag do we want to predict?
    tag_type = 'ner'

    # define the augment type
    augm = ocr_aug(corpus)

    # 2. make the tag  dictionary from the corpus
    tag_dictionary = augm.make_tag_dictionary(tag_type=tag_type)

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    from flair.embeddings import TransformerWordEmbeddings

    embeddings = TransformerWordEmbeddings(
        model='xlm-roberta-large',
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        context_dropout=0.,
    )

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    from flair.models import SequenceTagger

    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # 6. initialize trainer with AdamW optimizer
    from flair.trainers import ModelTrainer

    trainer = ModelTrainer(tagger, augm, optimizer=torch.optim.AdamW)

    # 7. run training with XLM parameters (20 epochs, small LR)
    from torch.optim.lr_scheduler import OneCycleLR

    trainer.train(f"resources/taggers/char_aug/{dataset_name}_ner-english-large_{seed}",
                  learning_rate=5.0e-6,
                  mini_batch_size=4,
                  mini_batch_chunk_size=1,
                  train_with_dev=True,
                  max_epochs=50,
                  scheduler=OneCycleLR,
                  embeddings_storage_mode='none',
                  weight_decay=0.,
                  shuffle=True,
                  )