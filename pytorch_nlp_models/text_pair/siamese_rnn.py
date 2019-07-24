import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SiameseGRU(nn.Module):
    def __init__(self, vocab_size,
                 emb_dim,
                 hidden_dim = 300,
                 dropout = 0.5,
                 bidirectional = True,
                 num_layers = 1,
                 emb_weights = None,
                 emb_static = False,
                 num_labels = 2,
                 batch_first = True
                ):
        super(SiameseGRU, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.emb_weights = emb_weights
        self.emb_static = emb_static
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.batch_first = True

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = 0)
        if self.emb_weights is not None:
            self.emb = nn.Embedding.from_pretrained(self.emb_weights, freeze = self.emb_static)

        self.sp_dropout = nn.Dropout2d(self.dropout_rate)

        self.rnn = nn.GRU(self.emb_dim, self.hidden_dim,
                          bidirectional = self.bidirectional,
                          dropout = (0 if self.num_layers < 2 else self.dropout_rate),
                          batch_first = self.batch_first
                         )

        self.out_dropout = nn.Dropout(self.dropout_rate)

        if self.bidirectional:
            fc_input_dim = self.hidden_dim * 2 * 3
        else:
            fc_input_dim = self.hidden_dim * 1 * 3

        self.fc = nn.Linear(fc_input_dim, self.num_labels)

    def _do_sp_dropout(self, out):
        out = out.permute(0, 2, 1) # B * C * T
        out = out.unsqueeze(dim = 3) # B * C * T * 1
        out = sel.sp_dropout(out1) # B * C * T * 1
        out = out.squeeze(dim = 3) # B * C * T
        out = out.permute(0, 2, 1) # B * T * C
        return out


    def forward(self, ids_1, ids_2, lens_1, lens_2):
        # 先对lens_1和lens_2排序
        sorted_lens_1, sorted_bidx_1 = torch.sort(lens_1, descending = True)
        sorted_lens_2, sorted_bidx_2 = torch.sort(lens_2, descending = True)
        sorted_ids_1 = ids_1[sorted_bidx_1,:]
        sorted_ids_2 = ids_2[sorted_bidx_2,:]

        out1 = self.emb(sorted_ids_1) # B * T * C
        out2 = self.emb(sorted_ids_2)

        out1 = self._do_sp_dropout(out1)
        out2 = self._do_sp_dropout(out2)

        packed_out1 = pack_padded_sequence(out1, sorted_lens_1, batch_first = self.batch_first)
        packed_out2 = pack_padded_sequence(out2, sorted_lens_2, batch_first = self.batch_first)
        packed_outs1, _ = self.rnn(packed_out1)
        packed_outs2, _ = self.rnn(packed_out2)
        outs1, _ = pad_packed_sequence(packed_outs1, batch_first = self.batch_first)
        outs2, _ = pad_packed_sequence(packed_outs2, batch_first = self.batch_first)

        # 恢复原本的顺序
        _, inv_sorted_bidx_1 = torch.sort(sorted_bidx_1)
        _, inv_sorted_bidx_2 = torch.sort(sorted_bidx_2)

        outs1 = outs1[inv_sorted_bidx_1, :, :]
        outs2 = outs2[inv_sorted_bidx_2, :, :]

        # 取最后一个状态作为句子的表示
        bs = outs1.shape[0]
        lasts1 = []
        lasts2 = []
        for i in range(bs):
            lasts1.append(outs1[i, lens_1[i] - 1,:])

        for i in range(bs):
            lasts2.append(outs2[i, lens_2[i] - 1,:])

        out1 = torch.stack(lasts1, dim = 0)
        out2 = torch.stack(lasts2, dim = 0)

        # merge
        merge1 = out1 + out2
        merge2 = torch.abs(out1 - out2)
        merge3 = out1 * out2
        merge = torch.cat([merge1, merge2, merge3], dim = 1)
        merge = self.out_dropout(merge)
        logits = self.fc(merge)
        return logits, out1, out2
