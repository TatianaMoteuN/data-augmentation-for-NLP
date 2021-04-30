# 1. get the corpus
import  flair
from flair.datasets import CONLL_03
from mapping import twitter_ner_mapped
from mapping import onto_ner_mapped
from mapping import wikigold_ner_mapped
from mapping import webpages_ner_mapped
from flair.models import SequenceTagger

# English
dataset_source_name = "conll3"
dataset_target_name = "conll3"
for seed in [1,2,3]:
    flair.set_seed(123)

    if dataset_target_name == "conll3":
        corpus = CONLL_03()
    elif dataset_target_name == "wikipedia":
        corpus = wikigold_ner_mapped()
    elif dataset_target_name == "webpages":
        corpus = webpages_ner_mapped()
    elif dataset_target_name == "twitter":
        corpus = twitter_ner_mapped()
    elif dataset_target_name == "onto_ner":
        corpus = onto_ner_mapped()

    flair.set_seed(seed)
    tagger: SequenceTagger = SequenceTagger.load(f'resources/taggers/{dataset_source_name}_glove_{seed}/best-model.pt')
    print(tagger.tag_type)


    # corpus



    result, score = tagger.evaluate(corpus.test, f'resources/{dataset_target_name}_glove_{seed}', mini_batch_size=32)
    print(result.detailed_results)
