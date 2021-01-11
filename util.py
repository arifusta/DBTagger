import re
import io as io
from keras import backend as K
import numpy as np


def get_key_vector(fname, keyword):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if (tokens[0] == keyword):
            data[tokens[0]] = map(float, tokens[1:])
            break
    return data


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split()
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def retrieve_data_and_tag(path):
    file = open(path, 'r')
    sentences, sentence_tags = [], []
    for line in file:
        # line = re.sub(r"\".*\"","\"xxx\"",line)
        line = re.sub(r"'", "", line)
        word_tag_list = line.split()
        words = []
        tags = []
        for w in word_tag_list:
            item = w.split('/')
            if (len(item[0]) == 0):
                continue

            words.append(item[0])
            if (len(item) < 2):
                print(path, item)
            tags.append(item[1])
        sentences.append(words)
        sentence_tags.append(tags)
        if (len(sentences) != len(sentence_tags)):
            print(line)
    return sentences, sentence_tags


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
        token_sequences.append(token_sequence)

    return token_sequences


def post_pad_sequence_list(sequence_list, maxlength):
    new_sequence_list = []
    sequence_length = len(sequence_list[0][0])
    list_of_zeros = [0.0] * sequence_length
    for sequence in sequence_list:
        for i in range(maxlength - len(sequence)):
            sequence.append(list_of_zeros)
        new_sequence_list.append(sequence)
    return new_sequence_list


def create_confusion_matrix(predicted_sequences, actual_sequences, tag2index):
    confusion_matrix = np.zeros((len(tag2index), len(tag2index)))
    for seq, sequence in enumerate(predicted_sequences):
        for tok, token in enumerate(sequence):
            confusion_matrix[tag2index[actual_sequences[seq][tok]]][tag2index[token]] += 1
    return confusion_matrix


def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy


def compute_accuracy(predicted_tag, predicted_db_tag, actual_tag, actual_db_tag):
    total_tag = 0
    total_table_tag = 0
    total_attr_tag = 0
    total_value_tag = 0
    correct_tag = 0
    correct_table_tag = 0
    correct_attr_tag = 0
    correct_value_tag = 0
    for p_t, p_db, a_t, a_db in zip(predicted_tag, predicted_db_tag, actual_tag, actual_db_tag):
        for pred_t, pred_db, actual_t, actual_db in zip(p_t, p_db, a_t, a_db):
            total_tag = total_tag + 1
            if (pred_t == actual_t and pred_db == actual_db):
                correct_tag = correct_tag + 1
                if (actual_t in ['TABLE', 'TABLEREF']):
                    correct_table_tag = correct_table_tag + 1
                elif (actual_t in ['ATTR', 'ATTRREF']):
                    correct_attr_tag = correct_attr_tag + 1
                elif (actual_t in ['VALUE']):
                    correct_value_tag = correct_value_tag + 1
            if (actual_t in ['TABLE', 'TABLEREF']):
                total_table_tag = total_table_tag + 1
            elif (actual_t in ['ATTR', 'ATTRREF']):
                total_attr_tag = total_attr_tag + 1
            elif (actual_t in ['VALUE']):
                total_value_tag = total_value_tag + 1
    print(correct_tag, total_tag, correct_table_tag, total_table_tag, correct_attr_tag, total_attr_tag,
          correct_value_tag, total_value_tag)
    return [correct_tag / total_tag, correct_table_tag / total_table_tag, correct_attr_tag / total_attr_tag,
            correct_value_tag / total_value_tag]


def keyword_mapping_accuracy(actual_tag, actual_db_tag, predicted_tag, predicted_db_tag, keys):
    accurate = 0
    total = 0
    for index in range(0, len(actual_tag)):
        correct_prediction = True
        value_exists = False
        for a_t, a_d_t, p_t, p_d_t in zip(actual_tag[index], actual_db_tag[index], predicted_tag[index],
                                          predicted_db_tag[index]):
            if a_t in keys and not (value_exists):
                value_exists = True
            if a_t in keys and (a_t != p_t or a_d_t != p_d_t):
                correct_prediction = False
        if value_exists:
            total += 1
            if correct_prediction:
                accurate += 1
    return accurate / total


def keyword_mapping_accuracy_two(actual_tag, actual_db_tag, predicted_tag, predicted_db_tag, keys):
    accurate = 0
    total = 0
    for index in range(0, len(actual_tag)):
        for a_t, a_d_t, p_t, p_d_t in zip(actual_tag[index], actual_db_tag[index], predicted_tag[index],
                                          predicted_db_tag[index]):
            if a_t in keys:
                total += 1
                if a_t == p_t and a_d_t == p_d_t:
                    accurate += 1
    return accurate / total


def post_process_predictions(predicted_type_tag, predicted_db_tag):
    for i in range(len(predicted_type_tag)):
        for j in range(1, len(predicted_type_tag[i]) - 1):
            if (predicted_type_tag[i][j - 1] == predicted_type_tag[i][j + 1] and predicted_db_tag[i][j - 1] ==
                    predicted_type_tag[i][j + 1] and predicted_type_tag[i][j] != predicted_type_tag[i][j - 1] and
                    predicted_db_tag[i][j] != predicted_db_tag[i][j - 1]):
                predicted_type_tag[i][j] = predicted_type_tag[i][j - 1]
                predicted_db_tag[i][j] = predicted_db_tag[i][j - 1]
