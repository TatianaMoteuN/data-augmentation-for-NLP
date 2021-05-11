from flair.data import Sentence, Corpus
from flair.datasets import CONLL_03, SentenceDataset

# load corpus
corpus = CONLL_03()

# make a list for augmented sentences
augmented_sentences = []

# go through all train and dev sentences
for sentence in corpus.train:
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    # for token in sentence:
    #     if token.text not in punc:
    #         print(token.text)


    # TODO: do data augmentation here
    augmented_sentence: Sentence = " ".join(token.text for token in sentence if token.text not in punc)
    #print(augmented_sentence)

    #append to augmented sentences
    augmented_sentences.append(augmented_sentence)


# make a new corpus with the augmented sentences
corpus = Corpus(train=SentenceDataset(augmented_sentences),
                dev=corpus.dev,
                test=corpus.test)