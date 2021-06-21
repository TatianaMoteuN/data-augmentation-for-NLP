from torch.optim.adam import Adam

from flair.data import Corpus
from flair.datasets import CONLL_03
from mapping import (
        twitter_ner_mapped,
        onto_ner_mapped,
        wikigold_ner_mapped,
        webpages_ner_mapped
    )

import flair
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
import nlpaug
from flair.data import  Token
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


# 1. get the corpus
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

    augm = ocr_aug(corpus)


    # 2. create the label dictionary
    label_dict = augm.make_label_dictionary()

    # 3. initialize transformer document embeddings (many models are available)
    document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune=True)

    # 4. create the text classifier
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

    # 5. initialize the text classifier trainer with Adam optimizer
    trainer = ModelTrainer(classifier, augm, optimizer=Adam)

    # 6. start the training
    trainer.train(f"resources/taggers/char_aug/{dataset_name}_ocr_bert-base-uncased_{seed}",
                  learning_rate=3e-5, # use very small learning rate
                  mini_batch_size=16,
                  mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
                  max_epochs=50, # terminate after 50 epochs
                  save_final_model= True,
                  shuffle= True,

                  )
    trainer.train()