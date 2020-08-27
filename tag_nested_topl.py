import pathlib

from utils.utils_data import load_data_from_corpus, save_train_valid_test_nested_ner
from utils.utils import tag_one_sentence


PATH = str(pathlib.Path().absolute())
RAW_DATA_PATH = PATH + '/raw_data'
RESULTS_PATH = PATH + '/results'
MAX_LEVEL = 5
GROUP_TAG = False


def form_data(sentence):
    temp_sentence = []
    for token in sentence:
        word = token[0]
        tag = token[1:]
        temp_sentence.append([[word], tag])
    return temp_sentence


if __name__ == '__main__':
    all_sentences = load_data_from_corpus(RAW_DATA_PATH)
    nested_data_conll_format = []
    overlapping = []

    for idx, sentence in enumerate(all_sentences):
        taged_sentence = tag_one_sentence(sentence, MAX_LEVEL, GROUP_TAG)
        if taged_sentence == -1:
            print(idx, end=', ')
            overlapping.append(idx)
            continue

        nested_data_conll_format.append(form_data(taged_sentence))

    with open(RESULTS_PATH + "/vistec_nested_ner_corpus.txt", 'w') as file:
        save_train_valid_test_nested_ner(file, nested_data_conll_format)
