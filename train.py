import os
import torch
import torch.nn as nn
import torch.optim as optim
import LSTM_project_ner.utils as utils
from LSTM_project_ner.model.model import LSTMTagger
from LSTM_project_ner.evaluate import evaluate
from random import shuffle
"""
TODO:
implementeer:   - dataset XXX niet gelukt
                - evaluate
                - opslaan en laden van model
                - metrics ( accuracy of beter nog F1 scores )  wordt wel iets berkend....
                - zorgen dat +- optimaal kan draaien voor GPU            

"""

def train(model, batch_size, train_data, loss_fn, optimizer, device):
    """
    Train network on a batch of data.
    """

    for i in range(batch_size):
        sentence, tags = train_data.pop()


        # Move the data over to the GPU
        input_sentence = sentence.to(device)
        targets = tags.to(device)

        model.zero_grad()

        tag_scores = model(input_sentence)

        loss = loss_fn(tag_scores, targets)
        loss.backward()
        optimizer.step()

        # if i % 2000 == 0:
        #
    print("loss while training: ", loss.item())

def train_and_evaluate(model, loss_fn, optimizer, epochs, train_data, val_data, batch_size, device):

    best_score = 0.0


    for epoch in range(epochs):

        model.train()

        # train on train data
        shuffle(train_data)
        train_data_batch = train_data.copy()
        num_steps = (len(train_data)) // batch_size
        for _ in range(num_steps):
            train(model, batch_size, train_data_batch, loss_fn, optimizer, device)

        # evaluate on val data
        # todo: vraag moet dit ook in batches?
        shuffle(val_data)
        val_data_batch = val_data.copy()
        # todo: voeg de laatste parameters samen tot een dict
        model.eval()
        eval_metrics = evaluate(model, val_data_batch, loss_fn, word_to_ix, label_to_ix, ix_to_label, device)

        # save model last
        torch.save(model.state_dict(), 'model/last_model.pt')

        cur_score = eval_metrics['f1-score']

        if cur_score > best_score:
            # als beter model save als best
            print("beter modolll gevondon")
            torch.save(model.state_dict(), 'model/best_model.pt')
            best_score = cur_score

        print(eval_metrics)

if __name__ == '__main__':

    print("------- prep data --------")
    # Load data
    path = 'data'
    sentence_path = os.path.join(path, 'train/sentences.txt')
    label_path = os.path.join(path, 'train/labels.txt')
    test_sentence_path = os.path.join(path, 'test/sentences.txt')
    test_label_path = os.path.join(path, 'test/labels.txt')
    val_sentence_path = os.path.join(path, 'val/sentences.txt')
    val_label_path = os.path.join(path, 'val/labels.txt')

    train_sentence = []
    train_labels = []

    test_sentence = []
    test_labels = []

    val_sentence = []
    val_labels = []

    utils.load_data(sentence_path, train_sentence)
    utils.load_data(label_path, train_labels)
    utils.load_data(test_sentence_path, test_sentence)
    utils.load_data(test_label_path, test_labels)
    utils.load_data(val_sentence_path, val_sentence)
    utils.load_data(val_label_path, val_labels)

    train_data = []
    test_data = []
    val_data = []

    for i in range(len(train_sentence)):
        train_data.append((train_sentence[i], train_labels[i]))

    for i in range(len(test_sentence)):
        test_data.append((test_sentence[i], test_labels[i]))

    for i in range(len(val_sentence)):
        val_data.append((val_sentence[i], val_labels[i]))

    word_to_ix = {}
    label_to_ix = {}

    utils.append_to_vocab(train_data, word_to_ix, label_to_ix)
    utils.append_to_vocab(test_data, word_to_ix, label_to_ix)
    utils.append_to_vocab(val_data, word_to_ix, label_to_ix)

    idx_train_data = []
    for sentence, tags in train_data:
        idx_sentences = utils.prepare_sequence(sentence, word_to_ix)
        idx_labels = utils.prepare_sequence(tags, label_to_ix)
        idx_train_data.append((idx_sentences, idx_labels))

    idx_test_data = []
    for sentence, tags in test_data:
        idx_sentences = utils.prepare_sequence(sentence, word_to_ix)
        idx_labels = utils.prepare_sequence(tags, label_to_ix)
        idx_test_data.append((idx_sentences, idx_labels))

    idx_val_data = []
    for sentence, tags in val_data:
        idx_sentences = utils.prepare_sequence(sentence, word_to_ix)
        idx_labels = utils.prepare_sequence(tags, label_to_ix)
        idx_val_data.append((idx_sentences, idx_labels))

    ix_to_label = {v: k for k, v in label_to_ix.items()}

    print("------- prep data done ------")

    # parameters for model
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(label_to_ix))
    loss_function = nn.NLLLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 5
    batch_size = 300

    # training
    train_and_evaluate(model, loss_function, optimizer, epochs, idx_train_data, idx_val_data, batch_size, device)

    model.load_state_dict(torch.load('model/best_model.pt'))
    # evaluate trained model
    evaluate(model, idx_test_data, loss_function, word_to_ix, label_to_ix, ix_to_label, device, report=True)
