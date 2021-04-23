from torch.optim.adam import Adam

from flair.data import Corpus
from flair.datasets import CONLL_03
from mapping import (
        twitter_ner_mapped,
        onto_ner_mapped,
        wikigold_ner_mapped
    )
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


# 1. get the corpus
corpus = CONLL_03()

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. initialize transformer document embeddings (many models are available)
document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', fine_tune=True)

# 4. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 5. initialize the text classifier trainer with Adam optimizer
trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

# 6. start the training
trainer.train('resources/taggers/trec/conll3',
              learning_rate=3e-5, # use very small learning rate
              mini_batch_size=16,
              mini_batch_chunk_size=4, # optionally set this if transformer is too much for your machine
              max_epochs=50, # terminate after 50 epochs
              save_final_model= True,
              shuffle= True,

              )
trainer.train()