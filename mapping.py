# -*- coding: utf-8 -*-
"""mapping.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jeq3BJAlO9c4cDIN0Zt9izrBM6uR4Fwr
"""
import flair
import logging
from abc import abstractmethod
from pathlib import Path

from flair.datasets import (
    TWITTER_NER,
    WIKIGOLD_NER,
    MIT_MOVIE_NER_SIMPLE,
    ColumnCorpus
)
from flair.data import Corpus

log = logging.getLogger("flair")

class Mapping():
    """
    A simple mapping object to map down the datasets into the conll_3 tag
    """

    def __init__(self, corpus, corpus_mapped):
        """
        Instantiate Mapping
        :param corpus: initial dataset
        :param corpus_mapped: dataset mapped into connl_4 tag set
        """
        self.corpus = corpus
        self.corpus_mapped = corpus_mapped


    def twitter_ner_mapped(self):
        self.corpus_mapped = TWITTER_NER(
    label_name_map={'facility': 'MISC','movie': 'MISC','product': 'MISC','sportsteam': 'ORG','tvshow': 'MISC','other': 'MISC','person': 'PER','geo-loc': 'LOC','company': 'ORG','band': 'ORG'}
    )
        return self.corpus_mapped


    def wikigold_ner_mapped(self):
        self.corpus = WIKIGOLD_NER()
        return self.corpus


    def onto_ner_mapped(self):
        self.corpus_mapped: Corpus = ColumnCorpus(
                "resources/tasks/onto-ner",
                column_format={0: "text", 1: "pos", 2: "upos", 3: "ner"},
                tag_to_bioes="ner",
                label_name_map={'NORP':'ORG','FAC':'LOC','GPE':'LOC','CARDINAL':'MISC','DATE':'MISC','EVENT':'MISC','LANGUAGE':'MISC','LAW':'MISC','MONEY':'MISC','ORDINAL':'MISC','PERCENT':'MISC','PRODUCT':'MISC','QUANTITY':'MISC','TIME':'MISC','WORK_OF_ART':'MISC'}
            )
        return self.corpus_mapped


    def mit_movie_ner_mapped(self):
        self.corpus_mapped = MIT_MOVIE_NER_SIMPLE(
    label_name_map={'SONG': 'MISC','PLOT': 'MISC','YEAR': 'MISC','GENRE': 'MISC','REVIEW': 'MISC','RATING': 'MISC','RATINGS_AVERAGE': 'MISC','TITLE': 'MISC','TRAILER': 'MISC','DIRECTOR': 'PER','ACTOR': 'PER','CHARACTER': 'PER'}
    )
        return self.corpus_mapped


    def mot_restaurant_ner_mapped(self):
        self.corpus_mapped = MIT_MOVIE_NER_SIMPLE(
    label_name_map={'SONG': 'MISC','PLOT': 'MISC','YEAR': 'MISC','GENRE': 'MISC','REVIEW': 'MISC','RATING': 'MISC','RATINGS_AVERAGE': 'MISC','TITLE': 'MISC','TRAILER': 'MISC','DIRECTOR': 'PER','ACTOR': 'PER','CHARACTER': 'PER'}
    )
        return self.corpus_mapped


