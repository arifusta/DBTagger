import os
import sys
#home = str(Path.home())
os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ ["CUDA_VISIBLE_DEVICES"] = sys.argv[6]

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Embedding, Activation, Reshape, Lambda, concatenate, RepeatVector, GRU
from keras.optimizers import Adam, Adadelta, Nadam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils import plot_model
from pathlib import Path
from util import *
from NEpochLogger import *
from ChainCRF import *
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import KFold, StratifiedKFold
from pympler import asizeof

root_path = '../..'
print (os.listdir(root_path))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


schema = sys.argv[1]
dropout = float(sys.argv[2])
epoch1  = int(sys.argv[3])
epoch2 = int(sys.argv[4])

if(sys.argv[5] == 'skip'):
    skip = True
else: # Uni-directional
    skip = False


if (schema == 'scholar' or schema == 'imdb' or schema == 'yelp'):
    train_sentences, train_pos_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TrainPos.txt')
    _, train_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TrainTag.txt')
    _, train_db_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TrainDbTag.txt')

    test_sentences, test_pos_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TestPos.txt')
    _, test_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TestTag.txt')
    _, test_db_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TestDbTag.txt')

    all_pos_tags = []
    all_pos_tags.extend(train_pos_tags)
    all_pos_tags.extend(test_pos_tags)

    all_tags = []
    all_tags.extend(train_tags)
    all_tags.extend(test_tags)

    all_db_tags = []
    all_db_tags.extend(train_db_tags)
    all_db_tags.extend(test_db_tags)

    all_sentences = train_sentences + test_sentences
else:
    all_sentences, all_pos_tags = retrieve_data_and_tag(root_path + '/Dataset/Spider/' + schema + '/questionsPOS.txt')
    _, all_tags = retrieve_data_and_tag(root_path + '/Dataset/Spider/' + schema + '/questionsTypeTag2.txt')
    _, all_db_tags = retrieve_data_and_tag(root_path + '/Dataset/Spider/' + schema + '/questionsDbTag2.txt')

#indexing words and tags
words, tags, db_tags, pos_tags = set([]), set([]), set([]), set([])

for ts in all_pos_tags:
    for t in ts:
        pos_tags.add(t)

for ts in all_tags:
    for t in ts:
        tags.add(t)

for ts in all_db_tags:
    for t in ts:
        db_tags.add(t)

word2embedding = load_vectors(root_path + '/Embeddings/tokens.vec')

tag2index = {t: i + 1 for i, t in enumerate(sorted(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

dbtag2index = {t: i + 1 for i, t in enumerate(sorted(db_tags))}
dbtag2index['-PAD-'] = 0  # The special value used to padding

postag2index = {t: i + 1 for i, t in enumerate(sorted(pos_tags))}
postag2index['-PAD-'] = 0  # The special value used to padding

#converting words and tags to indexes
all_sentences_X, all_tags_y, all_db_tags_y, all_pos_tags_y = [], [], [], []

#create input vectors using word representations
for idx, s in enumerate(all_sentences):
    s_int = []
    for w in s:
        try:
            word_vector = word2embedding[w]
            s_int.append(word_vector)
        except KeyError:
            print('Key error',w, 'bu ne',s)

    all_sentences_X.append(s_int)

#convert ouput labels into numeric categorical representations
for s in all_tags:
    all_tags_y.append([tag2index[t] for t in s])

for s in all_db_tags:
    all_db_tags_y.append([dbtag2index[t] for t in s])

for s in all_pos_tags:
    all_pos_tags_y.append([postag2index[t] for t in s])

#lenght of the longest sentence
MAX_LENGTH = len(max(all_sentences, key=len))
print("Schema is" + schema)
print('before padding max length =', MAX_LENGTH)

all_sentences_X = post_pad_sequence_list(all_sentences_X, MAX_LENGTH) #pad input first
all_tags_y = pad_sequences(all_tags_y, maxlen=MAX_LENGTH, padding='post') #pad_sequences tab output
all_db_tags_y = pad_sequences(all_db_tags_y, maxlen=MAX_LENGTH, padding='post') #pad_sequences db output
all_pos_tags_y = pad_sequences(all_pos_tags_y, maxlen=MAX_LENGTH, padding='post') #pad_sequences pos output
print('after padding length = ', len(all_sentences_X[0]))

#convert sentence embeddings to np.array
all_sentences_array = np.array(all_sentences_X)
print(all_sentences_array.shape)

if(skip):
    model_path = root_path + '/Models/MultiTask_WithSkip_' + sys.argv[1] + '_'
else: # Uni-directional
    model_path = root_path + '/Models/MultiTask_NoSkip_' + sys.argv[1] + '_'

#run variables initialized
totalRun = 0
kf = KFold(n_splits=6, shuffle=True, random_state=2)
fold_no = 2


fold_no = 1
avg_word_based_rel_acc = 0
avg_sent_based_rel_acc = 0
avg_word_based_nonrel_acc = 0
avg_sent_based_nonrel_acc = 0
avg_f1 = 0
avg_acc = 0
bs = 32
lr = 0.001

#task weights initialized
type_weight = 0.2
db_weight = 0.7
pos_weight = 0.1

for train,test in kf.split(all_sentences_array):

    #pos_weight = 1 - (db_weight + type_weight)

    #print(train)
    #print(test)
    out_epoch = NEpochLogger(display=20)
    totalRun = totalRun + 1

    # LSTM Model
    K.clear_session()
    # network model

    input_layer = Input((MAX_LENGTH, all_sentences_array[train].shape[2]))
    output_of_LSTM = Bidirectional(GRU(100, dropout=dropout, recurrent_dropout=dropout, return_sequences=True, unroll=True))(input_layer) #shared bi-lstm
    output_of_LSTM_pos_tag = GRU(100, dropout=dropout, recurrent_dropout=dropout, return_sequences=True,
                                 unroll=True)(output_of_LSTM) #unshared lstm for pos
    output_of_time_distributed_pos_tag = TimeDistributed(Dense(len(postag2index)))(output_of_LSTM_pos_tag)

    if (skip): ##Use skip connection
        #input_to_tag_LSTM = concatenate([output_of_LSTM, output_of_time_distributed_pos_tag], axis=2)
        output_of_LSTM_tag = GRU(100, dropout=dropout, recurrent_dropout=dropout, return_sequences=True,
                                 unroll=True)(output_of_LSTM)
        input_to_tag_dist = concatenate([output_of_LSTM_tag, output_of_time_distributed_pos_tag], axis=2)
        output_of_time_distributed_tag = TimeDistributed(Dense(len(tag2index)))(input_to_tag_dist)

        #input_to_db_tag_LSTM = concatenate([output_of_LSTM, output_of_time_distributed_pos_tag, output_of_time_distributed_tag], axis=2)
        output_of_LSTM_db_tag = GRU(100, dropout=dropout, recurrent_dropout=dropout,
                                    return_sequences=True, unroll=True)(output_of_LSTM)
        input_to_db_tag_dist = concatenate([output_of_LSTM_db_tag, output_of_time_distributed_tag], axis=2)
        output_of_time_distributed_db_tag = TimeDistributed(Dense(len(dbtag2index)))(input_to_db_tag_dist)
    else:
        output_of_LSTM_tag = GRU(100, dropout=dropout, recurrent_dropout=dropout, return_sequences=True,
                                 unroll=True)(output_of_LSTM) #unshared lstm for type
        output_of_time_distributed_tag = TimeDistributed(Dense(len(tag2index)))(output_of_LSTM_tag)
        output_of_LSTM_db_tag = GRU(100, dropout=dropout, recurrent_dropout=dropout,
                                    return_sequences=True, unroll=True)(output_of_LSTM) #unshared lstm for db
        output_of_time_distributed_db_tag = TimeDistributed(Dense(len(dbtag2index)))(output_of_LSTM_db_tag)

    # output_layer_tag = Activation('softmax')(output_of_time_distributed_tag)
    crf_tag = ChainCRF(name=sys.argv[1] + 'type_crf')
    output_layer_tag = crf_tag(output_of_time_distributed_tag)

    # output_layer_db_tag = Activation('softmax')(output_of_time_distributed_db_tag)
    crf_db_tag = ChainCRF(name=sys.argv[1] + 'db_crf')
    output_layer_db_tag = crf_db_tag(output_of_time_distributed_db_tag)

    # output_layer_pos_tag = Activation('softmax')(output_of_time_distributed_pos_tag)
    crf_pos_tag = ChainCRF(name=sys.argv[1] + 'pos_crf')
    output_layer_pos_tag = crf_pos_tag(output_of_time_distributed_pos_tag)
    model = Model(input_layer, [output_layer_tag, output_layer_db_tag, output_layer_pos_tag])

    checkpointer = ModelCheckpoint(model_path + 'fold' + str(fold_no) + '.h5',
                                   monitor='val_' + sys.argv[1] + 'db_crf_ignore_accuracy', verbose=0,
                                   save_best_only=True, save_weights_only=True, mode='max', period=1)

    earlystopper = EarlyStopping(monitor='val_' + sys.argv[1] + 'db_crf_ignore_accuracy', min_delta=0, patience=20, verbose=1,
                                 restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_' + sys.argv[1] + 'db_crf_ignore_accuracy', factor=0.5, mode='max',
                                  patience=5, min_lr=0.0001, min_delta=0)

    print('-------Training for Fold Run #' + str(fold_no) + '-------')
    model.compile(loss=[crf_tag.loss, crf_db_tag.loss, crf_pos_tag.loss],
                  loss_weights=[type_weight, db_weight, pos_weight], optimizer=Adadelta(clipnorm=1.),
                  metrics=['accuracy', ignore_class_accuracy(0)])

    model.summary()

    # training with AdaDelta first
    model.fit(all_sentences_array[train],
              [to_categorical(all_tags_y, len(tag2index))[train],
               to_categorical(all_db_tags_y, len(dbtag2index))[train],
               to_categorical(all_pos_tags_y, len(postag2index))[train]]
              , batch_size=bs, epochs=epoch1, shuffle=True, verbose=0, callbacks=[out_epoch],
              validation_split=0.2
              )

    model.compile(loss=[crf_tag.loss, crf_db_tag.loss, crf_pos_tag.loss],
                  loss_weights=[type_weight, db_weight, pos_weight], optimizer=Nadam(lr, clipnorm=1.),
                  metrics=['accuracy', ignore_class_accuracy(0)])

    # incremental training with Nadam
    model.fit(all_sentences_array[train],
              [to_categorical(all_tags_y, len(tag2index))[train],
               to_categorical(all_db_tags_y, len(dbtag2index))[train],
               to_categorical(all_pos_tags_y, len(postag2index))[train]]
              , batch_size=bs, epochs=epoch2, shuffle=True, verbose=0,
              callbacks=[out_epoch, checkpointer],
              validation_split=0.2
              )

    model.load_weights(model_path + 'fold' + str(fold_no) + '.h5')
    print("Size of DBTagger model: " + str (asizeof.asizeof(model)))
    predicted = model.predict(all_sentences_array[test])

    #first get prediction for type tags
    predictedtags = logits_to_tokens(predicted[0], {i: t for t, i in tag2index.items()})
    predicted_tags = []
    kfold_test_tags = []
    for i, sequence in zip(test, predictedtags):
        predicted_tags.append(sequence[:len(all_tags[i])])
        kfold_test_tags.append(all_tags[i])

    #then get prediction for db tags
    predicteddbtags = logits_to_tokens(predicted[1], {i: t for t, i in dbtag2index.items()})
    predicted_db_tags = []
    kfold_test_db_tags = []
    for i, sequence in zip(test, predicteddbtags):
        predicted_db_tags.append(sequence[:len(all_db_tags[i])])
        kfold_test_db_tags.append(all_db_tags[i])

    avg_word_based_rel_acc += keyword_mapping_accuracy_two(kfold_test_tags, kfold_test_db_tags, predicted_tags,
                                                       predicted_db_tags,
                                                       ['TABLE', 'ATTR', 'TABLEREF', 'ATTRREF'])
    avg_sent_based_rel_acc += keyword_mapping_accuracy(kfold_test_tags, kfold_test_db_tags, predicted_tags,
                                                   predicted_db_tags,
                                                   ['TABLE', 'ATTR', 'TABLEREF', 'ATTRREF'])

    avg_word_based_nonrel_acc += keyword_mapping_accuracy_two(kfold_test_tags, kfold_test_db_tags, predicted_tags,
                                                          predicted_db_tags, ['VALUE'])
    avg_sent_based_nonrel_acc += keyword_mapping_accuracy(kfold_test_tags, kfold_test_db_tags, predicted_tags,
                                                      predicted_db_tags, ['VALUE'])

    avg_acc += accuracy_score(kfold_test_db_tags, predicted_db_tags)
    avg_f1 += f1_score(kfold_test_db_tags, predicted_db_tags)

    fold_no = fold_no + 1

    with open(root_path + '/Results/dbTagger/dbTags_'+ schema + str(fold_no) + '.txt', 'w') as f:
        for prediction in predicted_db_tags:
            pred = ' '.join(prediction)
            print(pred + "\n", file=f)

    with open(root_path + '/Results/dbTagger/typeTags_'+ schema + str(fold_no) + '.txt', 'w') as f:
        for prediction in predicted_tags:
            pred = ' '.join(prediction)
            print(pred + "\n", file=f)

if (skip):
    outFileName = root_path + '/Results/MultiTask_WithSkip_AVG.txt'
else:
    outFileName = root_path + '/Results/MultiTask_NoSkip_AVG.txt'

with open(outFileName, 'a') as f:
    print('Schema: ' + sys.argv[1] + "\n", file=f)
    print('Dropout: ' + str(dropout) + "-BS:" + str(bs) + "\n", file=f)
    print('**************************', file=f)
    print("Accuracy:" + str(avg_acc / 6), file=f)
    print("F1-Score:" + str(avg_f1 / 6) + "\n", file=f)
    print('**************************', file=f)
    print('word based rel acc:',
          str (avg_word_based_rel_acc / 6), file=f)
    print('sentence based rel acc:',
          str (avg_sent_based_rel_acc / 6), file=f)
    print('word based non-rel acc:',
          str (avg_word_based_nonrel_acc / 6) , file=f)
    print('sentence based non-rel acc:',
          str (avg_sent_based_nonrel_acc / 6), file=f)
    print('**************************', file=f)