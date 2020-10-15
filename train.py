"""
Train
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from model.model import LSTMTagger, LSTMTagger_wPOS, double_LSTMTagger_wPOS
from evaluate import evaluate
from random import shuffle


def train(model, batch_size, train_data, loss_fn, optimizer, device, with_pos):
    """
    train aanvullen.

    :param model:
    :param batch_size:
    :param train_data:
    :param loss_fn:
    :param optimizer:
    :param device:
    :param with_pos:
    :return:
    """

    for i in range(batch_size):
        sentence, pos, tags = train_data.pop()
        # sentence, tags = train_data.pop()

        # Move the data over to the GPU
        input_sentence = sentence.to(device)
        input_pos = pos.to(device)
        targets = tags.to(device)

        model.zero_grad()

        if with_pos:
            tag_scores = model(input_sentence, input_pos)
        else:
            tag_scores = model(input_sentence)

        loss = loss_fn(tag_scores, targets)
        loss.backward()
        optimizer.step()

    print("loss while training: ", loss.item())

def train_and_evaluate(model, loss_fn, optimizer, epochs, train_data, val_data, batch_size, device, with_pos=False):
    """
    train and evaluate
    :param model:
    :param loss_fn:
    :param optimizer:
    :param epochs:
    :param train_data:
    :param val_data:
    :param batch_size:
    :param device:
    :param with_pos:
    :return:
    """

    best_score = 0.0

    for epoch in range(epochs):

        model.train()

        # train on train data
        shuffle(train_data)
        train_data_batch = train_data.copy()
        num_steps = (len(train_data)) // batch_size
        for _ in range(num_steps):
            train(model, batch_size, train_data_batch, loss_fn, optimizer, device, with_pos)

        # evaluate on val data
        shuffle(val_data)
        val_data_batch = val_data.copy()
        # todo: voeg de laatste parameters samen tot een dict
        model.eval()
        eval_metrics = evaluate(model, val_data_batch, loss_fn, word_to_ix, label_to_ix, ix_to_label, ix_to_word, device, with_pos)

        # save model last
        torch.save(model.state_dict(), 'model/last_model.pt')

        cur_score = eval_metrics['f1-score']

        if cur_score > best_score:
            # als beter model save als best
            print("beter modolll gevondon")
            torch.save(model.state_dict(), 'model/best_model.pt')
            best_score = cur_score

        print("epoch: ", epoch, eval_metrics)

if __name__ == '__main__':

    print("------- prep data --------")
    # Load data
    path = 'data/data_pos_ner_small/combined'
    sentence_path = os.path.join(path, 'train/sentences.txt')
    label_path = os.path.join(path, 'train/labels.txt')
    pos_path = os.path.join(path, 'train/pos.txt')

    test_sentence_path = os.path.join(path, 'test/sentences.txt')
    test_label_path = os.path.join(path, 'test/labels.txt')
    test_pos_path = os.path.join(path, 'test/pos.txt')


    val_sentence_path = os.path.join(path, 'val/sentences.txt')
    val_label_path = os.path.join(path, 'val/labels.txt')
    val_pos_path = os.path.join(path, 'val/pos.txt')

    train_sentence = []
    train_labels = []
    train_pos = []

    test_sentence = []
    test_labels = []
    test_pos = []

    val_sentence = []
    val_labels = []
    val_pos = []

    utils.load_data(sentence_path, train_sentence)
    utils.load_data(label_path, train_labels)
    utils.load_data(pos_path, train_pos)

    utils.load_data(test_sentence_path, test_sentence)
    utils.load_data(test_label_path, test_labels)
    utils.load_data(test_pos_path, test_pos)

    utils.load_data(val_sentence_path, val_sentence)
    utils.load_data(val_label_path, val_labels)
    utils.load_data(val_pos_path, val_pos)

    train_data = []
    test_data = []
    val_data = []

    for i in range(len(train_sentence)):
        train_data.append((train_sentence[i], train_pos[i], train_labels[i]))

    for i in range(len(test_sentence)):
        test_data.append((test_sentence[i], test_pos[i], test_labels[i]))

    for i in range(len(val_sentence)):
        val_data.append((val_sentence[i], val_pos[i], val_labels[i]))

    word_to_ix = {}
    label_to_ix = {}
    pos_to_ix = {}

    utils.append_to_vocab(train_data, word_to_ix, label_to_ix, pos_to_ix)
    utils.append_to_vocab(test_data, word_to_ix, label_to_ix, pos_to_ix)
    utils.append_to_vocab(val_data, word_to_ix, label_to_ix, pos_to_ix)

    idx_train_data = []
    for sentence, pos, tags in train_data:
        idx_sentences = utils.prepare_sequence(sentence, word_to_ix)
        idx_labels = utils.prepare_sequence(tags, label_to_ix)
        idx_pos = utils.prepare_sequence(pos, pos_to_ix)
        idx_train_data.append((idx_sentences, idx_pos, idx_labels))

    idx_test_data = []
    for sentence, pos, tags in test_data:
        idx_sentences = utils.prepare_sequence(sentence, word_to_ix)
        idx_labels = utils.prepare_sequence(tags, label_to_ix)
        idx_pos = utils.prepare_sequence(pos, pos_to_ix)
        idx_test_data.append((idx_sentences, idx_pos, idx_labels))

    idx_val_data = []
    for sentence, pos, tags in val_data:
        idx_sentences = utils.prepare_sequence(sentence, word_to_ix)
        idx_labels = utils.prepare_sequence(tags, label_to_ix)
        idx_pos = utils.prepare_sequence(pos, pos_to_ix)
        idx_val_data.append((idx_sentences, idx_pos, idx_labels))

    ix_to_label = {v: k for k, v in label_to_ix.items()}
    ix_to_word = {v: k for k, v in word_to_ix.items()}

    print("------- prep data done ------")

    # parameters for model
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 128
    pos_embedding_dim = 32
    word_embedding_dim = 20
    # model = LSTMTagger(WORD_EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(label_to_ix))
    model = LSTMTagger_wPOS(word_embedding_dim, pos_embedding_dim, HIDDEN_DIM, len(word_to_ix), len(pos_to_ix), len(label_to_ix))
    with_pos = True

    loss_function = nn.NLLLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 20
    batch_size = 5

    # training
    train_and_evaluate(model, loss_function, optimizer, epochs, idx_train_data, idx_val_data, batch_size, device, with_pos=with_pos)

    model.load_state_dict(torch.load('model/best_model.pt'))
    # evaluate trained model
    evaluate(model, idx_test_data, loss_function, word_to_ix, label_to_ix, ix_to_label, ix_to_word, device, with_pos=with_pos, report=True)
