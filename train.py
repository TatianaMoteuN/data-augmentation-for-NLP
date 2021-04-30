from typing import List

import flair
import flair.datasets
from flair.data import Corpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
)
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter

#get the corpus
#flair.set_seed(1)
import flair
# flair.set_seed(2)
# flair.set_seed(3)
from flair.datasets import CONLL_03
from mapping import (
        twitter_ner_mapped,
        onto_ner_mapped,
        wikigold_ner_mapped,
        webpages_ner_mapped
    )

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


    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings("glove"),
        # comment in this line to use character embeddings
        # CharacterEmbeddings(),
        # comment in these lines to use contextual string embeddings
        #
        # FlairEmbeddings('news-forward'),
        #
        # FlairEmbeddings('news-backward'),
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

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        f"resources/taggers/{dataset_name}_glove_{seed}",
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=100,
        shuffle=True,
    )

