# import os, glob, itertools
#
# def generate_collection(tag="train"):

with open('/home/aimsgh/SCIoI/data-augmentation-for-NLP/resources/tasks/webpages/webpages_ner.gold', 'r', errors='replace') as f:
    flag = None
    text = ""
    for line in f.readlines():

        l = line.strip()
        l = ' '.join(str(l).split())
        ls = l.split(" ")
        if len(ls) >= 2:
            # print(ls[5])
           word = ls[5]
           ner = ls[0]
           #print(word, ner)
           text += "\t".join([word, ner]) + '\n'
        else:
            text += '\n'

    text += '\n'

with open("/home/aimsgh/SCIoI/data-augmentation-for-NLP/resources/tasks/webpages/webpage" + ".ner", 'w') as f:
    f.write(text)

