"""
Build a vocabulary consisting of words and corresponding labels from a dataset

todo:

- kijk naar:
    import nltk
    nltk.download('punkt')
    for word in nltk.word_tokenize(s): #Tokenizing the words
"""


# additional parameters for vocab
import os

import utils

PAD = '<pad>'
UNK = '<UNK>'


def store_list_to_path(path, vocab, list_of_lists=False):
    """
    save
    """
    if list_of_lists:
        with open(path, "w") as f:
            for l in vocab:
                f.write(" ".join(l) + '\n')
    else:
        with open(path, "w") as f:
            for l in vocab:
                f.write(l + '\n')


def fill_vocab(path, vocab):
    """
    Fill vocabulary based on given path
    """
    # enumerate beetje overbodig
    with open(path) as f:
        for l in f:
            # print()
            # vocab.update()
            for w in l.strip().split(' '):
                vocab.add(w)


def preprocess_data(path, size):
    """

    :param path:
    :param size:
    :return:
    """
    preprocess_sentences = []
    preprocess_labels = []

    with open(path) as f:
        lines = f.read()

    raw_data = lines.split("\n")

    raw_sentence = []
    raw_labels = []

    for line in raw_data:
        split_line = line.split(" ")
        # only look at word/label pairs
        if len(split_line) > 1:
            current_word = split_line[0]
            current_label = split_line[1]
            # create separate sentence/label lists
            raw_sentence.append(current_word)
            raw_labels.append(current_label)
        else:
            # add sentence/label to corresponding list
            if 80 > len(raw_sentence) > 2:
                preprocess_sentences.append(raw_sentence)
                preprocess_labels.append(raw_labels)

            raw_sentence = []
            raw_labels = []
    return preprocess_sentences, preprocess_labels


def store_prossed_data(task_path, data_path, task_dir, sub_dir):
    """

    :param task_path:
    :param data_path:
    :param task_dir:
    :param sub_dir:
    :return:
    """

    path_task_sonar = os.path.join(task_path, task_dir)
    path_train = os.path.join(path_task_sonar, sub_dir + '_train')
    path_test = os.path.join(path_task_sonar, sub_dir + '_test')
    path_val = os.path.join(path_task_sonar, sub_dir + '_dev')

    preprocess_sentences_train, preprocess_labels_train = preprocess_data(path_train, 100)
    preprocess_sentences_test, preprocess_labels_test = preprocess_data(path_test, 20)
    preprocess_sentences_val, preprocess_labels_val = preprocess_data(path_val, 300)

    task_path_data = os.path.join(data_path, task_dir)
    path_sentence_train = os.path.join(task_path_data, 'train/sentences.txt')
    path_sentence_test = os.path.join(task_path_data, 'test/sentences.txt')
    path_sentence_val = os.path.join(task_path_data, 'val/sentences.txt')
    path_labels_train = os.path.join(task_path_data, 'train/labels.txt')
    path_labels_test = os.path.join(task_path_data, 'test/labels.txt')
    path_labels_val = os.path.join(task_path_data, 'val/labels.txt')

    store_list_to_path(path_sentence_train, preprocess_sentences_train, list_of_lists=True)
    store_list_to_path(path_sentence_test, preprocess_sentences_test, list_of_lists=True)
    store_list_to_path(path_sentence_val, preprocess_sentences_val, list_of_lists=True)
    store_list_to_path(path_labels_train, preprocess_labels_train, list_of_lists=True)
    store_list_to_path(path_labels_test, preprocess_labels_test, list_of_lists=True)
    store_list_to_path(path_labels_val, preprocess_labels_val, list_of_lists=True)

    print(task_dir, "data length: ")
    print(len(preprocess_sentences_train), len(preprocess_labels_train))
    print(len(preprocess_sentences_test), len(preprocess_labels_test))
    print(len(preprocess_sentences_val), len(preprocess_labels_val))

if __name__ == '__main__':

    # comment
    data_path = 'data/data_pos_srl'
    SONAR_path = 'data/SONAR/TASKS'

    store_prossed_data(SONAR_path, data_path, 'POS', 'pos_coarse')
    store_prossed_data(SONAR_path, data_path, 'SRL', 'srl_main')

    data_path_ner = os.path.join(data_path, 'SRL')
    data_path_pos = os.path.join(data_path, 'POS')

    ner_train_sentence_path = os.path.join(data_path_ner, 'train/sentences.txt')
    ner_train_label_path = os.path.join(data_path_ner, 'train/labels.txt')
    pos_train_sentence_path = os.path.join(data_path_pos, 'train/sentences.txt')
    pos_train_label_path = os.path.join(data_path_pos, 'train/labels.txt')

    ner_test_sentence_path = os.path.join(data_path_ner, 'test/sentences.txt')
    ner_test_label_path = os.path.join(data_path_ner, 'test/labels.txt')
    pos_test_sentence_path = os.path.join(data_path_pos, 'test/sentences.txt')
    pos_test_label_path = os.path.join(data_path_pos, 'test/labels.txt')

    ner_val_sentence_path = os.path.join(data_path_ner, 'val/sentences.txt')
    ner_val_label_path = os.path.join(data_path_ner, 'val/labels.txt')
    pos_val_sentence_path = os.path.join(data_path_pos, 'val/sentences.txt')
    pos_val_label_path = os.path.join(data_path_pos, 'val/labels.txt')


    ner_train_sentence = []
    ner_train_labels = []
    pos_train_sentence = []
    pos_train_labels = []

    utils.load_data(ner_train_sentence_path, ner_train_sentence, as_list=False)
    utils.load_data(ner_train_label_path, ner_train_labels, as_list=False)
    utils.load_data(pos_train_sentence_path, pos_train_sentence, as_list=False)
    utils.load_data(pos_train_label_path, pos_train_labels, as_list=False)

    utils.load_data(ner_test_sentence_path, ner_train_sentence, as_list=False)
    utils.load_data(ner_test_label_path, ner_train_labels, as_list=False)
    utils.load_data(pos_test_sentence_path, pos_train_sentence, as_list=False)
    utils.load_data(pos_test_label_path, pos_train_labels, as_list=False)

    utils.load_data(ner_val_sentence_path, ner_train_sentence, as_list=False)
    utils.load_data(ner_val_label_path, ner_train_labels, as_list=False)
    utils.load_data(pos_val_sentence_path, pos_train_sentence, as_list=False)
    utils.load_data(pos_val_label_path, pos_train_labels, as_list=False)

    # utils.load_data(test_sentence_path, test_sentence)
    # utils.load_data(test_label_path, test_labels)
    # utils.load_data(val_sentence_path, val_sentence)
    # utils.load_data(val_label_path, val_labels)

    d_pos_train = dict()
    d_ner_train = dict()
    d_combi = dict()
    count = 0

    # train
    for i in range(len(pos_train_sentence)):
        if pos_train_sentence[i] not in d_pos_train:
            d_pos_train[pos_train_sentence[i]] = pos_train_labels[i]

    for i in range(len(ner_train_sentence)):
        if ner_train_sentence[i] not in d_ner_train:
            d_ner_train[ner_train_sentence[i]] = ner_train_labels[i]

    for s in d_pos_train.keys():
        if s in d_ner_train:
            count += 1
            d_combi[s] = (d_pos_train[s], d_ner_train[s])

    sentences = []
    labels = []
    pos = []

    for s, (p, l) in d_combi.items():
        sentences.append(s)
        labels.append(l)
        pos.append(p)

    assert len(sentences) == len(labels) == len(pos)

    split_1 = int(len(sentences) * 0.7)
    split_2 = split_1 + int(len(sentences) * 0.15)

    train_sentences = sentences[:split_1]
    test_sentences = sentences[split_1:split_2]
    val_sentences = sentences[split_2:]
    print(len(train_sentences) + len(test_sentences) + len(val_sentences))

    train_labels = labels[:split_1]
    test_labels = labels[split_1:split_2]
    val_labels = labels[split_2:]

    train_pos = pos[:split_1]
    test_pos = pos[split_1:split_2]
    val_pos = pos[split_2:]

    task_path_data = os.path.join(data_path, 'combined')
    path_sentence_train = os.path.join(task_path_data, 'train/sentences.txt')
    path_sentence_test = os.path.join(task_path_data, 'test/sentences.txt')
    path_sentence_val = os.path.join(task_path_data, 'val/sentences.txt')

    path_labels_train = os.path.join(task_path_data, 'train/labels.txt')
    path_labels_test = os.path.join(task_path_data, 'test/labels.txt')
    path_labels_val = os.path.join(task_path_data, 'val/labels.txt')

    path_pos_train = os.path.join(task_path_data, 'train/pos.txt')
    path_pos_test = os.path.join(task_path_data, 'test/pos.txt')
    path_pos_val = os.path.join(task_path_data, 'val/pos.txt')

    train_x = 50
    valtest_x = 10
    store_list_to_path(path_sentence_train, train_sentences[:train_x])
    store_list_to_path(path_sentence_test, test_sentences[:valtest_x])
    store_list_to_path(path_sentence_val, val_sentences[:valtest_x])
    store_list_to_path(path_labels_train, train_labels[:train_x])
    store_list_to_path(path_labels_test, test_labels[:valtest_x])
    store_list_to_path(path_labels_val, val_labels[:valtest_x])
    store_list_to_path(path_pos_train, train_pos[:train_x])
    store_list_to_path(path_pos_test, test_pos[:valtest_x])
    store_list_to_path(path_pos_val, val_pos[:valtest_x])
