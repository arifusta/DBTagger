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
import os
from pathlib import Path
from util import *
from NEpochLogger import *
from ChainCRF import *
import sys
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import KFold, StratifiedKFold

root_path = '../..'
print (os.listdir(root_path))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#home = str(Path.home())
os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ ["CUDA_VISIBLE_DEVICES"] = '0'

schema = sys.argv[1]
dropout = float(sys.argv[2])
epoch1  = int(sys.argv[3])
epoch2 = int(sys.argv[4])

if(sys.argv[5] == 'bi'):
    bi = True
else: # Uni-directional
    bi = False


if (schema == 'scholar' or schema == 'imdb' or schema == 'yelp'):
    #train_sentences, train_pos_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TrainPos.txt')
    #_, train_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TrainTag.txt')
    train_sentences, train_db_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TrainDbTag.txt')

    #test_sentences, test_pos_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TestPos.txt')
    #_, test_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TestTag.txt')
    test_sentences, test_db_tags = retrieve_data_and_tag(root_path + '/FixedDataset/' + schema + 'TestDbTag.txt')

    #all_pos_tags = []
    #all_pos_tags.extend(train_pos_tags)
    #all_pos_tags.extend(test_pos_tags)

    #all_tags = []
    #all_tags.extend(train_tags)
    #all_tags.extend(test_tags)

    all_db_tags = []
    all_db_tags.extend(train_db_tags)
    all_db_tags.extend(test_db_tags)

    all_sentences = train_sentences + test_sentences
else:
    #all_sentences, all_pos_tags = retrieve_data_and_tag(root_path + '/FixedDataset/Spider/' + schema + '/questionsPOS.txt')
    #_, all_tags = retrieve_data_and_tag(root_path + '/FixedDataset/Spider/' + schema + '/questionsTypeTag2.txt')
    all_sentences, all_db_tags = retrieve_data_and_tag(root_path + '/FixedDataset/Spider/' + schema + '/questionsDbTag2.txt')

# indexing words and tags
words, db_tags= set([]), set([])

for ts in all_db_tags:
    for t in ts:
        db_tags.add(t)

dbtag2index = {t: i + 1 for i, t in enumerate(sorted(db_tags))}
dbtag2index['-PAD-'] = 0  # The special value used to padding

# word2index = {w: i + 2 for i, w in enumerate(list(words))}
# word2index['-PAD-'] = 0  # The special value used for padding
# word2index['-OOV-'] = 1  # The special value used for OOVs
word2embedding = load_vectors(root_path + '/Embeddings/tokens.vec')
all_sentences_X = []
all_db_tags_Y = []

for idx, s in enumerate(all_sentences):
    s_int = []
    for w in s:
        try:
            word_vector = word2embedding[w]
            s_int.append(word_vector)
        except KeyError:
            print('Key error',w, 'bu ne',s)

    all_sentences_X.append(s_int)

for s in all_db_tags:
    all_db_tags_Y.append([dbtag2index[t] for t in s])

#lenght of the longest sentence
MAX_LENGTH = len(max(all_sentences, key=len))
print("Schema is" + schema)
print('before padding max length =', MAX_LENGTH)

all_sentences_X = post_pad_sequence_list(all_sentences_X, MAX_LENGTH) #pad input first
all_db_tags_y = pad_sequences(all_db_tags_Y, maxlen=MAX_LENGTH, padding='post') #pad_sequences output
print('after padding length = ', len(all_sentences_X[0]))

#convert sentence embeddings to np.array
all_sentences_array = np.array(all_sentences_X)
print(all_sentences_array.shape)

#f1 score to be used in metrics in fit
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if(bi):
    model_path = root_path + '/Models/SingleTask_Bi_' + sys.argv[1] + '_'
else: # Uni-directional
    model_path = root_path + '/Models/SingleTask_Uni_' + sys.argv[1] + '_'

#run variables initialized
totalRun = 0
kf = KFold(n_splits=6, shuffle=True, random_state=2)
fold_no = 1
word_based_rel_acc = 0
sent_based_rel_acc = 0
word_based_nonrel_acc = 0
sent_based_nonrel_acc = 0
avg_f1 = 0
avg_acc = 0
bs = 32
lr = 0.001

for train,test in kf.split(all_sentences_array):
    out_epoch = NEpochLogger(display=20)
    totalRun = totalRun + 1

    # LSTM Model
    K.clear_session()
    input_layer = Input((MAX_LENGTH, all_sentences_array[train].shape[2]))

    if (bi):
        shared_LSTM = Bidirectional(GRU(100, dropout = dropout, recurrent_dropout= dropout, return_sequences=True, unroll=True))(input_layer)
    else:  # Uni-directional
        shared_LSTM = GRU(100, dropout = dropout, recurrent_dropout= dropout, return_sequences=True, unroll=True)(input_layer)

    shared_LSTM2 = GRU(100, dropout = dropout, recurrent_dropout= dropout, return_sequences=True, unroll=True)(shared_LSTM)
    td_dbtag= TimeDistributed(Dense(len(dbtag2index)))(shared_LSTM2)
    crf_db_tag = ChainCRF(name=sys.argv[1] + 'db_crf')
    output_layer_db_tag = crf_db_tag(td_dbtag)

    model = Model(input_layer, output_layer_db_tag)
    checkpointer = ModelCheckpoint(model_path + 'fold' + str(fold_no) + '.h5',
                                   monitor='val_ignore_accuracy', verbose=0,
                                   save_best_only=True, save_weights_only=True, mode='max', period=1)

    earlystopper = EarlyStopping(monitor='val_ignore_accuracy', min_delta=0, patience=20, verbose=1,
                                 restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_ignore_accuracy', factor=0.5, mode='max',
                                      patience=5, min_lr=0.0001, min_delta=0)

    print('-------Training for Fold Run #' + str(fold_no) + '-------')
    # print('Weights: ' + weightString)
    # print('Batchsize: ' + str(bs))
    model.compile(loss=[crf_db_tag.loss],
                  optimizer=Adadelta(clipnorm=1.),
                  metrics=['accuracy', ignore_class_accuracy(0)])

    model.summary()
    # training with AdaDelta first
    model.fit(all_sentences_array[train], to_categorical(all_db_tags_y, len(dbtag2index))[train]
              ,batch_size=bs, epochs=epoch1, shuffle=True, verbose=0, callbacks=[out_epoch],
              validation_split=0.2
              )

    model.compile(loss=[crf_db_tag.loss],
                  optimizer=Nadam(lr, clipnorm=1.),
                  metrics=['accuracy', ignore_class_accuracy(0)])

    # incremental training with Nadam
    model.fit(all_sentences_array[train], to_categorical(all_db_tags_y, len(dbtag2index))[train]
              , batch_size=bs, epochs=epoch2, shuffle=True, verbose=0, callbacks=[out_epoch, checkpointer, earlystopper, reduce_lr],
              validation_split=0.2
              )

    model.load_weights(model_path + 'fold' + str(fold_no) + '.h5')
    scores = model.evaluate(all_sentences_array[test], to_categorical(all_db_tags_y, len(dbtag2index))[test])

    print(model.metrics_names)
    print(f"{model.metrics_names[1]}: {scores[1] * 100}")
    print(f"{model.metrics_names[2]}: {scores[2] * 100}")

    # start_time = time.clock()
    predicted = model.predict(all_sentences_array[test])
    # print("Execution time all: ", time.clock() - start_time, "seconds")

    predicteddbtags = logits_to_tokens(predicted, {i: t for t, i in dbtag2index.items()})

    predicted_db_tags = []
    kfold_test_db_tags = []
    for i, sequence in zip(test, predicteddbtags):
        predicted_db_tags.append(sequence[:len(all_db_tags[i])])
        kfold_test_db_tags.append(all_db_tags[i])

    avg_acc += accuracy_score(kfold_test_db_tags, predicted_db_tags)
    avg_f1 += f1_score(kfold_test_db_tags, predicted_db_tags)

    fold_no = fold_no + 1

if (bi):
    outFileName = root_path + '/Results/SingleTask_Bi_AVG.txt'
else:
    outFileName = root_path + '/Results/SingleTask_Uni_AVG.txt'

with open(outFileName, 'a') as f:
    print('Schema: ' + sys.argv[1] + "\n", file=f)
    print("Accuracy:" + str(avg_acc / 6), file=f)
    # print("Precision:" + str(precision_score(kfold_test_db_tags, predicted_db_tags)), file=f)
    # print("Recall:" + str(recall_score(kfold_test_db_tags, predicted_db_tags)), file=f)
    print("F1-Score:" + str(avg_f1 / 6) + "\n", file=f)