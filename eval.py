# 1. get the corpus
from flair.datasets import CONLL_03
from mapping import twitter_ner_mapped
from mapping import onto_ner_mapped
from mapping import wikigold_ner_mapped
from flair.models import SequenceTagger

# English
tagger: SequenceTagger = SequenceTagger.load('resources/taggers/best-model.pt')
print(tagger.tag_type)


# corpus

corpus = CONLL_03()

result, score = tagger.evaluate(corpus.test, 'resources/conll3', mini_batch_size=32)
print(result.detailed_results)