import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class LSTMTagger_wPOS(nn.Module):

    def __init__(self, word_embedding_dim, pos_embedding_dim, hidden_dim, vocab_size, pos_size, tagset_size):
        super(LSTMTagger_wPOS, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_size, pos_embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(word_embedding_dim + pos_embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, pos):

        # combineer pos en word embed
        word_embeds = self.word_embeddings(sentence)
        pos_embeds = self.pos_embeddings(pos)

        cat_embeds = torch.cat((word_embeds, pos_embeds),dim=1)

        lstm_out, _ = self.lstm(cat_embeds.view(len(sentence), 1, -1))

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class double_LSTMTagger_wPOS(nn.Module):

    def __init__(self, word_embedding_dim, pos_embedding_dim, hidden_dim, vocab_size, pos_size, tagset_size):
        super(double_LSTMTagger_wPOS, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_size, pos_embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        self.lstm_word = nn.LSTM(word_embedding_dim, hidden_dim)
        self.lstm_pos = nn.LSTM(pos_embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, pos):

        # combineer pos en word embed
        word_embeds = self.word_embeddings(sentence)
        pos_embeds = self.pos_embeddings(pos)


        lstm_out_word, _ = self.lstm_word(word_embeds.view(len(sentence), 1, -1))
        lstm_out_pos, _ = self.lstm_pos(pos_embeds.view(len(sentence), 1, -1))

        tag_space_word = self.hidden2tag(lstm_out_word.view(len(sentence), -1))
        tag_space_pos = self.hidden2tag(lstm_out_pos.view(len(sentence), -1))

        out = (tag_space_word + tag_space_pos) * 0.5

        tag_scores = F.log_softmax(out, dim=1)
        return tag_scores