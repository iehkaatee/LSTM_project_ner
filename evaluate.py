"""
    evaluate function
"""
import torch
import warnings

from sklearn_crfsuite.metrics import flat_classification_report

from LSTM_project_ner import utils

#
# def f1_score_without_warning(all_tags, all_pred_tags):
#     return flat_classification_report(all_tags, all_pred_tags, labels=label_list, digits=3, zero_division='warn',
#                                       output_dict=True)['weighted avg']['f1-score']

def evaluate(model, eval_data, loss_fn, word_to_ix, label_to_ix, ix_to_label, device, report=False):
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

    for sentence, tags in eval_data:

        #inputs = utils.prepare_sequence(sentence, word_to_ix)
        #inputs = inputs.to('cpu')

        # Move the data over to the GPU
        input_sentence = sentence.to(device)
        targets = tags.to(device)

        label_score = model(input_sentence)
        loss = loss_fn(label_score, targets)

        predictions = [torch.max(x, 0)[1].item() for x in label_score]

        #correct = utils.prepare_sequence(tags, label_to_ix)

        original_sentence = sentence
        correct_labels = [ix_to_label[t] for t in tags.tolist()]
        predicted_labels = [ix_to_label[p] for p in predictions]
        all_tags.append(correct_labels)
        all_pred_tags.append(predicted_labels)

        # print(flat_classification_report(all_tags, all_pred_tags, labels=label_list , digits=3, zero_division='warn'))
        # x = flat_classification_report(all_tags, all_pred_tags, labels=label_list , digits=3, zero_division='warn', output_dict=True)['weighted avg']['f1-score']

        # analog sanity metrics
        for i in range(len(tags)):
            total += 1
            if predicted_labels[i] == correct_labels[i]:
                number_correct += 1
                if correct_labels[i] != "O":
                    ner_correct += 1
            if correct_labels[i] != "O":
                total_loc += 1

        if report:
            print('{:15}|{:15}|{:15}'.format(*['original', 'correct', 'predicted']))
            for item in zip(original_sentence, correct_labels, predicted_labels):
                print('{:<15}|{:<15}|{:<15}'.format(*item))
            print()



    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_metrics = flat_classification_report(all_tags, all_pred_tags, labels=label_list, digits=3, output_dict=True)['weighted avg']

        if report:
            print(flat_classification_report(all_tags, all_pred_tags, labels=label_list, digits=3))
            print(f"{number_correct} of {total} guessed correct of all labels")
            print(f"{ner_correct} of {total_loc} guessed correct of ner labels")

    eval_metrics['loss'] = loss.item()
    eval_metrics['ner_correct'] = ner_correct / float(total_loc)
    return eval_metrics
