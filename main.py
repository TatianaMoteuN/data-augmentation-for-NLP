# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


from flair.datasets import CONLL_03
from mapping import (
        twitter_ner_mapped,
        onto_ner_mapped,
        wikigold_ner_mapped
    )
corpus = onto_ner_mapped()

print(corpus.train[6].to_tagged_string('ner'))
