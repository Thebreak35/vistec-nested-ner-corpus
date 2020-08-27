import re
import json
import os
from collections import deque

import numpy as np
from pythainlp.tokenize import word_tokenize

from .utils_for_test_data import check_overlapping


def update_idx(idx_en, shift_list):
    for shift, l, u in shift_list:
        if idx_en in list(range(l, u)):
            return shift
    return 0


def shift_index(entities, shift_list):
    entities = json.dumps(entities)
    entities = json.loads(entities)
    
    for idx, entity in enumerate(entities):
        entities[idx]['type'] = '_'.join(entity['type'].strip().split())
        entities[idx]['start_idx'] -= update_idx(entity['start_idx'], shift_list)
        entities[idx]['end_idx'] -= update_idx(entity['end_idx'], shift_list)
    return entities


def remove_pipe(entities, ssg_text):
    # Get index 
    list_idx = [idx for idx, c in enumerate(ssg_text) if c == '|']
    list_idx.append(len(ssg_text))
    list_pair_idx = [(i+1, list_idx[i], list_idx[i+1]+1) for i in range(len(list_idx)-1)]

    return shift_index(entities, list_pair_idx)


def get_all_entities_index(entities):
    entities = sorted(entities, key=lambda x: (x['start_idx'], -x['end_idx']))
    set_s = {element['start_idx'] for element in entities}
    set_e = {element['end_idx'] for element in entities}
    return set.union(set_s, set_e)


def space_tokenizer(words_tokenize):
    input_len = len(''.join(words_tokenize))
    
    # space tokenize 
    token_space = []
    for txt in words_tokenize:
        split_txt = []
        if txt.isspace():
            split_txt = txt
        else:
            split_txt = deque(re.split(r'(\s+)', txt))
            if txt[0] == ' ':
                split_txt = split_txt.popleft()
                
            if txt[-1] == ' ':
                split_txt.pop()
                
        token_space.extend(split_txt)
    
    output_len = len(''.join(token_space))
    assert output_len == input_len, "space_tokenizer error"
    return token_space
    
    
def token_en2words(set_index, txt_test_data, engine='newmm'):
    # Dumps and load json data
    txt_test_data = json.dumps(txt_test_data)
    txt_test_data = json.loads(txt_test_data)
    temp = 0
    save_token = []
    
    for idx, _ in enumerate(range(len(txt_test_data))):
        # Tokenize entities
        if idx in set_index:
            e_token = txt_test_data[temp:idx]
            # Check empty text
            if len(e_token) > 0:
                # word tokenize
                # should rename variable 'words_tokenize'
                words_tokenize = []
                if engine in ['newmm', 'deepcut', 'attacut']:
                    words_tokenize = word_tokenize(e_token, engine=engine)
                elif engine is 'bpe':
                    pass
                else:
                    raise Exception('Tokenizer mismatch')
                words_tokenize = space_tokenizer(words_tokenize)             
                save_token.extend(words_tokenize)
                temp = idx
    
    words_tokenize = word_tokenize(txt_test_data[temp:], engine=engine)
    words_tokenize = space_tokenizer(words_tokenize) 
    save_token.extend(words_tokenize)
    return save_token


def tags_nested(words_token, entities, max_levels, group=False):
    entities = sorted(entities, key=lambda x: (x['start_idx'], -x['end_idx']))
    start_idx_ens = {e['start_idx'] for e in entities}
    end_idx_ens = {e['end_idx'] for e in entities}
    words_tags = []
    dict_entities = {}
    end_word_idx = 0
    
    for idx_word, word in enumerate(words_token):
        start_word_idx = end_word_idx
        end_word_idx = start_word_idx + len(word)
        
        # Push an entity into a dictionary
        # if the start entity equal to start
        # words token character index.
        if start_word_idx in start_idx_ens:
            match_start_idx = np.where(np.array(start_idx_ens) == start_word_idx)[0]
            for idx_m in match_start_idx:
                type_en_math = entities[idx_m]['type']
                type_en_math = '_'.join(type_en_math.split('-'))
                dict_entities.update({idx_m: type_en_math})
                
        if group:
            # Assign the nested tags for each word.
            words_tags.append((word, [grouptags(stk_en) for stk_en in dict_entities.items()]))

        else:
            # Assign the nested tags for each word.
            words_tags.append((word, [stk_en for stk_en in dict_entities.items()]))
        
        # Pop the entity into a dictionary 
        # when the end entity equal to end words 
        # token character index.
        if end_word_idx in end_idx_ens:
            match_end_idx = np.where(np.array(end_idx_ens) == end_word_idx)[0]
            for idx_m in match_end_idx:
                type_en_math = entities[idx_m]['type']
                try:
                    del dict_entities[idx_m]
                except Exception as e:
                    raise e
        else:
            pass
    
    nested_tags = []
    # Add O tags
    for word in words_tags:
        w = str(word[0])
        nested_tag = []
        for i in range(max_levels):
            try:
                tag = np.array(word[1])[i][0] + '-' + np.array(word[1])[i][1]
            except Exception as e:
                tag = 'O'
                raise e
                
            nested_tag.append(tag)

        nested_tags.append((w, '|'.join(nested_tag)))
    return nested_tags


def tags_bioes(words_tags, max_levels):
    words_tags.insert(0, ('start', '|'.join(['X' for x in range(max_levels)])))
    words_tags.insert(len(words_tags), ('end',  '|'.join(['X' for x in range(max_levels)])))
    
    nested_tags_sent = []
    for wi in range(len(words_tags) - 2):
        temp_word = []
        word = words_tags[wi + 1][0]
        
        # Replace ' ' by '_'
        if word.isspace():
            word = '_'
        
        temp_word.append(word)
        
        for wl in range(max_levels): 
            t1 = words_tags[wi][1].split('|')[wl]
            t2 = words_tags[wi + 1][1].split('|')[wl]
            t3 = words_tags[wi + 2][1].split('|')[wl]
            tag = tags_entity(t1, t2, t3)

            if tag is not 'O':
                tag = tag+'-' + t2.split('-')[1]

            temp_word.append(tag)

        nested_tags_sent.append(temp_word)

    words_tags.pop(0)
    words_tags.pop(-1)
    return nested_tags_sent


def transitionl1(t1, t2):
    x1 = '*'
    if t2 == 'O':
        x1 = 'O'
    elif t2 == 'X':
        x1 = 'X'
    elif t1 != t2:
        x1 = 'S'
    elif t1 == t2:
        x1 = 'I'
    else:
        print('Error !!! transitionl1 : 1')
    return x1


def transitionl2(x1, x2):
    x = '**'
    if x1 == 'S':
        if x2 in ['S', 'O', 'X']:
            x = 'S'
        elif x2 == 'I':
            x = 'B'
        else:
            print('Error !!! transitionl2 : 1')
    elif x1 == 'I':
        if x2 in ['S', 'O', 'X']:
            x = 'E'
        elif x2 == 'I':
            x = 'I'
        else:
            print('Error !!! transitionl2 : 2')
    elif x1 == 'O':
        x = 'O'
    elif x1 == 'X':
        x = 'X'
    else:
        print('Error !!! transitionl2 : 3')
    return x
    
    
def tags_entity(t1, t2, t3):
    x1 = transitionl1(t1, t2)
    x2 = transitionl1(t2, t3)
    return transitionl2(x1, x2)


def save_data(data, file, nested=False):
    for line in data:
        if nested:
            line = ' '.join(line)
        file.writelines(str(line)+'\n')
    file.writelines('\n')
    
    
def grouptags(tag):
    maintags = '''{
                    "Person"       : ["person", "title", "firstname", 
                                      "middle", "last", "nicknametitle", 
                                      "nickname", "namemod", "psudoname",
                                      "role"],
                                      
                    "Location"     : ["restaurant", "continent", "country" , 
                                      "state", "city" , "district", 
                                      "province", "sub_district", 
                                      "roadname" , "address", "soi", 
                                      "latitude", "longtitude" , "postcode", 
                                      "ocean", "island", "mountian", 
                                      "river", "space", "loc:others"],
                                      
                    "Time"         : ["date", "year", "month", 
                                      "day", "time", "duration", 
                                      "periodic" , "season", "rel"],
                                      
                    "Organisation" : ["orgcorp", "org:edu", "org:political", 
                                      "org:religious", "org:other", "goverment", 
                                      "army", "sports_team", "media", 
                                      "hotel", "museum", "hospital", 
                                      "band", "jargon", "stock_exchange", 
                                      "index", "fund"],
                                      
                    "NORP"         : ["nationality", "religion", "norp:political", 
                                      "norp:others"],
                                      
                    "Facility"     : ["airport", "port", "bridge", 
                                      "building", "stadium", "station", 
                                      "facility:other"],
                                      
                    "Event"        : ["sports_event","concert","natural_disaster",
                                      "war","event:others"],
                                      
                    "WOA"          : ["book", "film", "song", 
                                      "tv_show", "woa"],
                                      
                    "MISC"         : ["animate", "game", "language", 
                                      "law", "award", "electronics", 
                                      "weapon", "vehicle", "disease", 
                                      "god", "sciname", "food:ingredient", 
                                      "product:food", "product:drug", "animal_species"],
                                      
                    "NUM"          : ["cardinal" , "mult", "fold", 
                                      "money" , "energy", "speed", 
                                      "distance", "weight", "quantity", 
                                      "percent", "temperature", "unit"]
                }'''
    
    maintag = 'UNK'
    maintags = json.loads(maintags)
    idx_tag = list(tag)[0]
    tag = list(tag)[1]
    
    # Mapping sub-tags to main-tags
    for main in maintags.items():
        if tag in main[1]:
            maintag = main[0].lower()
            break
        elif tag is 'O':
            maintag = tag
            break
        else:
            continue
    
    if maintag is 'UNK':
        print(tag, "Error !!!: The tag doesn't match the main-tags.")
    return tuple(idx_tag, maintag)


def create_save_dir(path):
    # Create parent directory
    try:
        os.mkdir(path)
    except Exception as e:
        raise e("Exit directory")
        
    try:
        os.mkdir(path + '/flatten_ner')
        os.mkdir(path + '/nested_ner')
        os.mkdir(path + '/flatten_ner/subtags')
        os.mkdir(path + '/flatten_ner/maintags')
        os.mkdir(path + '/nested_ner/subtags')
        os.mkdir(path + '/nested_ner/maintags')
        os.mkdir(path + '/flatten_ner/subtags/datas')
        os.mkdir(path + '/flatten_ner/maintags/datas')
        os.mkdir(path + '/nested_ner/subtags/datas')
        os.mkdir(path + '/nested_ner/maintags/datas')
    except Exception as e:
        raise e("Exit directory")


def tag_one_sentence(tag_sentences, max_level, group):
    tag_sentences['entities'] = remove_pipe(tag_sentences['entities'], tag_sentences['ssg'])
    token_idx_en = get_all_entities_index(tag_sentences['entities'])
    words_token = token_en2words(token_idx_en, tag_sentences['text'], engine='newmm')
    words_tags_nested_ = tags_nested(words_token, tag_sentences['entities'], max_level, group=group)
    words_tags_nested = tags_bioes(words_tags_nested_, max_level)

    # Check overlapping entities:
    if check_overlapping(words_tags_nested_):
        return -1
    return words_tags_nested
