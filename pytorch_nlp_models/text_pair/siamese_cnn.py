import torch
import torch.nn as nn


class SiameseCNN(nn.Module):
    def __init__(self, vocab_size,
                 emb_dim,
                 hidden_dim = 300,
                 kernel_sizes = [1,3,5],
                 dropout = 0.5,
                 emb_weights = None,
                 emb_static = False,
                 num_labels = 2,
                ):
        super(SiameseCNN, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.emb_weights = emb_weights
        self.emb_static = emb_static
        self.num_labels = num_labels
        self.kernel_sizes = kernel_sizes

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx = 0)
        if self.emb_weights is not None:
            self.emb = nn.Embedding.from_pretrained(self.emb_weights, freeze = self.emb_static)

        self.sp_dropout = nn.Dropout2d(self.dropout_rate)
        self.cnns = nn.ModuleList([nn.Conv1d(self.emb_dim, self.hidden_dim, ks, padding = (ks - 1) // 2) for ks in self.kernel_sizes])

        self.out_dropout = nn.Dropout(self.dropout_rate)

        fc_input_dim = self.hidden_dim * len(self.kernel_sizes) * 3

        self.fc = nn.Linear(fc_input_dim, self.num_labels)

    def _do_sp_dropout(self, out):
        out = out.permute(0, 2, 1) # B * C * T
        out = out.unsqueeze(dim = 3) # B * C * T * 1
        out = self.sp_dropout(out) # B * C * T * 1
        out = out.squeeze(dim = 3) # B * C * T
        out = out.permute(0, 2, 1) # B * T * C
        return out


    def _do_cnns(self, out, length):
        out = out.permute(0, 2, 1) # B*C*T
        outs = []
        for cnn in self.cnns:
            _out = cnn(out)
            _out = _out.permute(0, 2, 1) # B*T*C
            _maxes = []
            for i in range(_out.shape[0]):
                _max, _ = torch.max(_out[i, :length[i], :], dim = 0) # C
                _maxes.append(_max)
            _out = torch.stack(_maxes, dim = 0) # B*C
            outs.append(_out)
        return torch.cat(outs, dim = 1)

    def forward(self, ids_1, ids_2, lens_1, lens_2):
        out1 = self.emb(ids_1) # B * T * C
        out2 = self.emb(ids_2)

        out1 = self._do_sp_dropout(out1)
        out2 = self._do_sp_dropout(out2)

        out1 = self._do_cnns(out1, lens_1)
        out2 = self._do_cnns(out2, lens_2)

        # merge
        merge1 = out1 + out2
        merge2 = torch.abs(out1 - out2)
        merge3 = out1 * out2
        merge = torch.cat([merge1, merge2, merge3], dim = 1)
        merge = self.out_dropout(merge)
        logits = self.fc(merge)
        return logits, out1, out2
