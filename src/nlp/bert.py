import torch
import torch.nn as nn
import gensim
from nlp.bert_embeddings import bert_embeddings
import pdb
torch.backends.cudnn.enabled = False

class BERTEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(BERTEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.bert = bert_embeddings()

        self.dec = nn.LSTM(input_size=1536,
                           hidden_size=512,
                           dropout=0.2,
                           batch_first=True).double()

    def sort(self, x, reverse=False):
        return zip(*sorted([(x[i], i) for i in range(len(x))], reverse=reverse))

    def sortNpermute(self, x, mask):
        mask_sorted, perm = self.sort(
            mask.sum(dim=-1).cpu().numpy(), reverse=True)
        return x[list(perm)], list(mask_sorted), list(perm)

    def inverse_sortNpermute(self, x, perm):
        _, iperm = self.sort(perm, reverse=False)
        if isinstance(x, list):
            return [x_[list(iperm)] for x_ in x]
        else:
            return x[list(iperm)]

    def forward(self, sentences):
        sentences = [x_.split(' ') for x_ in sentences]
        x_orig, mask_orig = self.bert.get_vectors(sentences)
        #print(x_orig.shape)
        # x_ = self.lin(x_orig)
        # print(x_.shape)
        ''' Here you will find why sort and permute is done
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence'''
        x, mask, perm = self.sortNpermute(x_orig, mask_orig)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, mask, batch_first=True)

        ''' forward pass through lstm '''
        x, (h, m) = self.dec(x)

        ''' get the output at time_step=t '''
        h = self.inverse_sortNpermute(h[-1], perm)

        # print(h1.shape)

        return h, x_orig
