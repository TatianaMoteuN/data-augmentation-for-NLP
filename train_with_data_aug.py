import flair
import flair.datasets
import nlpaug

from flair.data import Sentence, Corpus, Token
from flair.datasets import CONLL_03, SentenceDataset
from typing import List

from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
)

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

from flair.training_utils import EvaluationMetric

flair.device = 'cuda:0'

# get the corpus
from mapping import (
        twitter_ner_mapped,
        onto_ner_mapped,
        wikigold_ner_mapped,
        webpages_ner_mapped
    )

# load corpus
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
    tag_type = "ner"

    # define the augment type
    augm = ocr_aug(corpus)

    # 3. make the tag dictionary from the corpus
    tag_dictionary = augm.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings("glove"),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
    )

    # initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, augm)

    trainer.train(
        f"resources/taggers/char_aug/{dataset_name}_ocr_glove_{seed}",
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=50,
        shuffle=True,
    )