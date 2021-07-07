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
import nlpaug
from flair.datasets import CONLL_03
# from mapping import (
#         twitter_ner_mapped,
#         onto_ner_mapped,
#         wikigold_ner_mapped
#     )

from nlpaugment import(
    ocr_aug,
    keyboard_aug,
    random_insert_aug,
    random_subtitute_aug,
    random_swap_aug,
    random_delete_aug
)
corpus = CONLL_03()
print(corpus)
augm = random_delete_aug(corpus)
print(augm)
print(augm.train[40])
#corpus = onto_ner_mapped()

#print(corpus.train[6].to_tagged_string('ner'))



