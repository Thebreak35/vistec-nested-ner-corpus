import glob
import json
from collections import Counter

import pandas as pd


def sort_dict_freq(counter_dict, MINCOUNT=1, reverse=True):
    my_dict = [(w, c) for w, c in counter_dict.items() if c >= 1]
    my_dict = sorted(my_dict, key=lambda item: item[1], reverse=reverse)
    return [w[0] for w in my_dict]


def load_conll_format_nested_ner(path, max_level=5):
    corpus = []
    words = Counter()
    tags = Counter()
    chars = Counter()
    sentence = []
    with open(path) as file:
        for index, line in enumerate(file):
            line = line.strip('\n')
            token = line.split()
            len_sample = len(token)

            if len_sample == max_level + 1:
                word = token[0]
                tag = token[1:]
                
                words.update([word])
                chars.update(list(word))
                tags.update(tag)
                sentence.append([[word], tag])
            
            elif len_sample == 0 and line == '':
                corpus.append(sentence)
                sentence = []
        
        if len(sentence) == 0:
            pass
        else:
            corpus.append(sentence)

    return corpus, sort_dict_freq(words), sort_dict_freq(chars), sort_dict_freq(tags)


def load_data_from_corpus(path_corpus):
    # Load all data from raw_data/
    format_csv = '/**/*.csv'
    format_json = '/**/*.json'

    filenames_csv = [f for f in glob.glob(path_corpus + format_csv, recursive=True)]
    filename_json = [f for f in glob.glob(path_corpus + format_json, recursive=True)]

    for f in filenames_csv:
        print(f)
    for f in filename_json:
        print(f)

    all_sentences = []
    for filen_csv, filen_json in zip(filenames_csv, filename_json):
        # Check correction of the directory name and set file name
        checklotdir_csv = filen_csv.split('/')[-2]
        checklotdir_json = filen_json.split('/')[-2]
        assert checklotdir_csv == checklotdir_json, "Filename not match."

        # Load CSV file
        data_from_csv = pd.read_csv(filen_csv)

        # Load Json file 
        data_from_json = json.load(open(filen_json, "r"))

        # Tags entities each sentence
        for idx, (entities, ssg_text, cleaned_text) in enumerate(
                zip(data_from_json, data_from_csv['ssg'], data_from_csv['text_clean'])):
            sample = {
                'text': cleaned_text, 'ssg': ssg_text,
                'json_text': entities['text'], 'entities': entities['entities']
            }
            all_sentences.append(sample)
    return all_sentences


def save_train_valid_test_nested_ner(file, data):
    for sent in data:
        for token in sent:
            word, tags = token
            file.writelines(' '.join(word + tags) + '\n')
        file.writelines('\n')
