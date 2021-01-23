import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMTagger(nn.Module):
    """
        Basic LSTM used for only word data
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class LSTMTagger_wPOS(nn.Module):
    """
        LSTM that concatenates POS and word embedding layer to process data that is supplemented with POS
    """
    def __init__(self, word_embedding_dim, pos_embedding_dim, hidden_dim, vocab_size, pos_size, tagset_size):
        super(LSTMTagger_wPOS, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_size, pos_embedding_dim)

        self.lstm = nn.LSTM(word_embedding_dim + pos_embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, pos):

        word_embeds = self.word_embeddings(sentence)
        pos_embeds = self.pos_embeddings(pos)

        cat_embeds = torch.cat((word_embeds, pos_embeds),dim=1)

        lstm_out, _ = self.lstm(cat_embeds.view(len(sentence), 1, -1))

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
