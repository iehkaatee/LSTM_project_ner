"""
    evaluate function
"""

import os
from random import shuffle
import torch
import warnings
from sklearn_crfsuite.metrics import flat_classification_report
import utils
from conlleval import evaluate as con_eval
from model.model import LSTMTagger


def evaluate(model, eval_data, loss_fn, word_to_ix, label_to_ix, ix_to_label, ix_to_word, device='cpu', with_pos=False, report=False, show_sentence=False):
    """
        eval model
    """

    model.eval()
    ner_correct = 0
    number_correct = 0
    total = 0
    total_loc = 0
    all_tags = []
    all_pred_tags = []
    label_list = [k for k, _ in label_to_ix.items()]
    shuffle(eval_data)
    for sentence, pos, tags in eval_data:

        # Move the data over to the GPU
        input_sentence = sentence.to(device)
        input_pos = pos.to(device)
        targets = tags.to(device)

        if with_pos:
            label_score = model(input_sentence, input_pos)
        else:
            label_score = model(input_sentence)

        loss = loss_fn(label_score, targets)

        predictions = [torch.max(x, 0)[1].item() for x in label_score]

        # correct = utils.prepare_sequence(tags, label_to_ix)

        original_sentence = [ix_to_word[s] for s in input_sentence.tolist()]
        correct_labels = [ix_to_label[t] for t in targets.tolist()]
        predicted_labels = [ix_to_label[p] for p in predictions]
        all_tags.append(correct_labels)
        all_pred_tags.append(predicted_labels)

        # analog sanity metrics
        for i in range(len(tags)):
            total += 1
            if predicted_labels[i] == correct_labels[i]:
                number_correct += 1
                if correct_labels[i] != "O":
                    ner_correct += 1
            if correct_labels[i] != "O":
                total_loc += 1

        if report and show_sentence:
            print('{:30}|{:15}|{:15}'.format(*['original', 'correct', 'predicted']))
            for item in zip(original_sentence, correct_labels, predicted_labels):
                print('{:<30}|{:<15}|{:<15}'.format(*item))
            print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_metrics = flat_classification_report(all_tags, all_pred_tags, digits=3, output_dict=True)['weighted avg']

        if report:
            # Con eval standard evaluate
            all_tag_list = [tag for tag_list in all_tags for tag in tag_list]
            all_pred_list = [tag for tag_list in all_pred_tags for tag in tag_list]
            con_eval(all_tag_list, all_pred_list, verbose=True)
            print()

            # crf sklearn evaluate
            print("table with 'O'-token: ")
            print(flat_classification_report(all_tags, all_pred_tags, labels=label_list, digits=3))

            label_list.remove("O")
            print("table WITHOUT 'O'-token: ")
            print(flat_classification_report(all_tags, all_pred_tags, labels=label_list, digits=3))

            # custom evaluate
            print(f"{number_correct} of {total} guessed correct of all labels, {number_correct/total}% correct")
            print(f"{ner_correct} of {total_loc} guessed correct of ner labels, {ner_correct/total_loc}% correct")

    eval_metrics['loss'] = loss.item()
    eval_metrics['ner_correct'] = ner_correct / float(total_loc)
    return eval_metrics


if __name__ == '__main__':

    print("------- prep data --------")
    # Load data
    path = 'data/data_pos_ner/combined'
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

    print("------- prep data done ------\n")
    # parameters for model
    WORD_EMBEDDING_DIM = 64
    POS_EMBEDDING_DIM = 20
    HIDDEN_DIM = 48
    model = LSTMTagger(WORD_EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(label_to_ix))
    loss_function = torch.nn.NLLLoss()

    model.load_state_dict(torch.load('model/best_model_20201012_without_pos.pt',
                                     map_location=torch.device('cpu')))
    # evaluate trained model
    evaluate(model, idx_test_data, loss_function, word_to_ix, label_to_ix, ix_to_label, ix_to_word, report=True)



