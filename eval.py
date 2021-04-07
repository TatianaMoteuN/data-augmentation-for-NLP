# 1. get the corpus
from mapping import twitter_ner_mapped
from flair.models import SequenceTagger

# English
tagger: SequenceTagger = SequenceTagger.load('ner-fast')
print(tagger.tag_type)


# corpus

corpus = twitter_ner_mapped()

result, score = tagger.evaluate(corpus.test, mini_batch_size=32)
print(result.detailed_results)