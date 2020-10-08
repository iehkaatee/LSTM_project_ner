"""
Build a vocabulary consisting of words and corresponding labels from a dataset

todo:

- kijk naar:
    import nltk
    nltk.download('punkt')
    for word in nltk.word_tokenize(s): #Tokenizing the words
"""


# additional parameters for vocab
PAD = '<pad>'
UNK = '<UNK>'

def store_list_to_path(path, vocab, list_of_lists = False):
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

def preprocess_data(path, sentences, labels, size):
    """
    clean up raw data:
        - remove reference id
        - gather sentences/labels
    """

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
                sentences.append(raw_sentence)
                labels.append(raw_labels)

            raw_sentence = []
            raw_labels = []


if __name__ == '__main__':

    # comment
    preprocess_sentences_train = []
    preprocess_sentences_test = []
    preprocess_sentences_val = []
    preprocess_labels_train = []
    preprocess_labels_test = []
    preprocess_labels_val = []

    preprocess_data('SONAR/TASKS/NER/ner_train', preprocess_sentences_train, preprocess_labels_train,1000)
    preprocess_data('SONAR/TASKS/NER/ner_test', preprocess_sentences_test, preprocess_labels_test,300)
    preprocess_data('SONAR/TASKS/NER/ner_dev', preprocess_sentences_val, preprocess_labels_val, 300)

    store_list_to_path('data/train/sentences.txt', preprocess_sentences_train, list_of_lists=True)
    store_list_to_path('data/train/labels.txt', preprocess_labels_train, list_of_lists=True)
    store_list_to_path('data/test/sentences.txt', preprocess_sentences_test, list_of_lists=True)
    store_list_to_path('data/test/labels.txt', preprocess_labels_test, list_of_lists=True)
    store_list_to_path('data/val/sentences.txt', preprocess_sentences_val, list_of_lists=True)
    store_list_to_path('data/val/labels.txt', preprocess_labels_val, list_of_lists=True)

    print(len(preprocess_sentences_train), len(preprocess_labels_train))
    print(len(preprocess_sentences_test), len(preprocess_labels_test))
    print(len(preprocess_sentences_val), len(preprocess_labels_val))

    # create words dictionary
    words = set()
    train_sentences =   fill_vocab('data/train/sentences.txt', words)
    test_sentences =    fill_vocab('data/test/sentences.txt', words)
    dev_sentences =     fill_vocab('data/val/sentences.txt', words)

    # create labels dictionary
    labels = set()
    train_labels =      fill_vocab('data/train/labels.txt', labels)
    test_labels =       fill_vocab('data/test/labels.txt', labels)
    dev_labels =        fill_vocab('data/val/labels.txt', labels)

    # words.add(PAD), words.add(UNK)
    # labels.add(PAD), labels.add(UNK)
    #
    # # todo
    # words = list(words)
    # labels = list(labels)
    #
    # store_list_to_path('ner_experiment/data_small/words.txt', words)
    # store_list_to_path('ner_experiment/data_small/labels.txt', labels)
