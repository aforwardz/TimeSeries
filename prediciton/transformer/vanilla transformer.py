import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split, Dataset


# Transformer utils
class TransformerDataSet(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(TransformerDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    # assert seq_q.dim() == 2 and seq_k.dim() == 2
    seq_q_use = torch.sum(seq_q, dim=-1)
    seq_k_use = torch.sum(seq_k, dim=-1)
    batch_size, len_q = seq_q_use.size()
    batch_size, len_k = seq_k_use.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k_use.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    seq_use = torch.sum(seq, dim=-1)
    attn_shape = [seq_use.size(0), seq_use.size(1), seq_use.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask    # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        d_k = Q.shape[2]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

    def forward(self, input_Q, input_K, input_V, attn_mask):
        n_heads = self.n_heads
        d_k = self.d_k
        d_v = self.d_v
        d_model = self.d_model
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_v(=len_k), d_k]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        d_model = self.d_model
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs same to Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, src_feature_size, d_model, n_layers, d_k, d_v, n_heads, d_ff):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(src_feature_size, d_model, bias=False)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, tgt_feature_size, d_model, n_layers, d_k, d_v, n_heads, d_ff):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Linear(tgt_feature_size, d_model, bias=False)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_feature_size, tgt_feature_size, d_model, n_layers, d_k, d_v, n_heads, d_ff):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_feature_size, d_model, n_layers, d_k, d_v, n_heads, d_ff)
        self.decoder = Decoder(tgt_feature_size, d_model, n_layers, d_k, d_v, n_heads, d_ff)
        self.projection = nn.Linear(d_model, tgt_feature_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_logits
        dec_logits = self.projection(dec_outputs)
        return dec_logits


class TransformerPredictor:
    @classmethod
    def predict(cls, pred_data, nums, features, aux_info=None, pred_preiod=1, **kwargs):
        torch.manual_seed(0)
        np.random.seed(0)
        aux_info_dims = 0
        if aux_info is not None:
            aux_info_dims = aux_info.shape[-1]
            pred_data = np.hstack((pred_data, aux_info))

        train_ratio = 0.8
        pred_data, low, high = cls().data_normalize(pred_data)
        train_len = int(len(pred_data) * train_ratio)
        val_len = len(pred_data) - train_len
        pred_data_train = pred_data[:train_len]
        pred_data_val = pred_data[train_len:]

        kwargs['d_model'] = 128 if 'd_model' not in kwargs else kwargs['d_model']
        kwargs['n_layers'] = 6 if 'n_layers' not in kwargs else kwargs['n_layers']
        kwargs['n_heads'] = 8 if 'n_heads' not in kwargs else kwargs['n_heads']
        kwargs['d_ff'] = 512 if 'd_ff' not in kwargs else kwargs['d_ff']
        kwargs['epochs'] = 20 if 'epochs' not in kwargs else kwargs['epochs']
        kwargs['learning_rate'] = 0.0001 if 'learning_rate' not in kwargs else kwargs['learning_rate']
        kwargs['use_seq2seq'] = False if 'use_seq2seq' not in kwargs else kwargs['use_seq2seq']

        look_back = 5
        enc_inputs_train, dec_inputs_train, dec_outputs_train = cls().generate_tfm_data(pred_data_train, look_back,
                                                                                        pred_preiod, seq2seq=kwargs['use_seq2seq'])
        enc_inputs_val, dec_inputs_val, dec_outputs_val = cls().generate_tfm_data(pred_data_val, look_back,
                                                                                  pred_preiod, seq2seq=kwargs['use_seq2seq'])

        train_set = TransformerDataSet(enc_inputs_train, dec_inputs_train, dec_outputs_train)
        val_set = TransformerDataSet(enc_inputs_val, dec_inputs_val, dec_outputs_val)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        model = Transformer(src_feature_size=features + aux_info_dims,
                            tgt_feature_size=features + aux_info_dims,
                            d_model=kwargs['d_model'],
                            n_layers=kwargs['n_layers'],
                            d_k=64,
                            d_v=64,
                            n_heads=kwargs['n_heads'],
                            d_ff=kwargs['d_ff']
                            )
        criterion = nn.MSELoss()
        optimizer = SGD(model.parameters(), lr=kwargs['learning_rate'], momentum=0.9)

        trained_model = cls().tfm_train(train_loader, val_loader, model,
                                        criterion, optimizer, epochs=kwargs['epochs'], early_stop=False)
        if kwargs['use_seq2seq']:
            preds = cls().seq2seq_prediction(pred_data, trained_model, look_back, pred_preiod)
        else:
            preds = cls().rolling_prediction(pred_data, trained_model, look_back, pred_preiod)

        preds = cls().data_denormalize(preds, low, high)
        preds = preds[:, :features]

        return preds

    def tfm_train(self, train_loader, val_loader, model, criterion, optimizer, epochs=20,
                  verbose=True, save_model=False, early_stop=True):
        best_loss = 1000
        early_stop_count = 0
        for i in range(epochs):
            epoch_loss_train = self.train_one_epoch(train_loader, model, criterion, optimizer)
            epoch_loss_val = self.val_one_epoch(val_loader, model, criterion)

            if verbose:
                print("epoch: {} train loss: {}".format(i + 1, epoch_loss_train))
                print("epoch: {} val loss: {}".format(i + 1, epoch_loss_val))

            if epoch_loss_val < best_loss:
                early_stop_count = 0
                best_loss = epoch_loss_val
                if save_model:
                    model_name = "./best_transformer_model.pth"
                    torch.save(model.state_dict(), model_name)
            early_stop_count += 1
            if early_stop_count >= 10 and early_stop:
                break

        return model

    def train_one_epoch(self, train_loader, model, criterion, optimizer):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            enc_inputs, dec_inputs, dec_outputs = batch
            optimizer.zero_grad()
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def val_one_epoch(self, val_loader, model, criterion):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                enc_inputs, dec_inputs, dec_outputs = batch
                outputs = model(enc_inputs, dec_inputs)
                loss = criterion(outputs, dec_outputs)
                epoch_loss += loss.item()

        return epoch_loss / len(val_loader)

    def seq2seq_prediction(self, data, model, look_back, pred_period):
        pred_data = torch.tensor(data).float()
        pred_data = pred_data[-1 * look_back:]
        enc_input = pred_data.unsqueeze(dim=0)
        enc_outputs, enc_self_attns = model.encoder(enc_input)
        dec_input = torch.zeros((1, 1, enc_input.shape[2]))
        for i in range(pred_period + 1):
            dec_output, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
            dec_logits = model.projection(dec_output)
            dec_input = torch.cat((dec_input, dec_logits[:, -1, :].unsqueeze(1)), dim=1)

        preds = dec_logits[:, :pred_period, :].squeeze(0)
        return preds.detach().numpy()

    def rolling_prediction(self, data, model, look_back, pred_period):
        pred_data = torch.tensor(data).float()
        pred_data = pred_data[-1 * look_back:]
        preds = []

        for i in range(pred_period):
            enc_input = pred_data.unsqueeze(dim=0)
            pred = self.predict_word_by_word(enc_input, model)
            preds.append(pred.squeeze(0))
            temp_data = torch.cat((pred_data, pred))
            pred_data = temp_data[-1 * look_back]

        preds = torch.stack(preds)
        return preds.detach().numpy()

    def predict_word_by_word(self, enc_input, model):
        enc_outputs, enc_self_attns = model.encoder(enc_input)
        dec_input = torch.zeros((1, 1, enc_input.shape[2]))
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        dec_logits = model.projection(dec_outputs)

        return dec_logits.squeeze(0)

    def generate_tfm_data(self, data, look_back=10, pred_period=1, seq2seq=False):
        forward_step = pred_period if seq2seq else 1
        data = torch.tensor(data).float()
        nums, features = data.shape
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for i in range(nums - look_back - forward_step):
            enc_input = data[i:i + look_back]
            dec_input = torch.cat((torch.zeros((1, features)), data[i + look_back: i + look_back + forward_step]))
            dec_output = torch.cat((data[i + look_back: i + look_back + forward_step], torch.zeros((1, features))))
            enc_inputs.append(enc_input)
            dec_inputs.append(dec_input)
            dec_outputs.append(dec_output)
        enc_inputs = torch.stack(enc_inputs)
        dec_inputs = torch.stack(dec_inputs)
        dec_outputs = torch.stack(dec_outputs)

        return enc_inputs.float(), dec_inputs.float(), dec_outputs.float()

    def data_normalize(self, pred_data):
        low = np.min(pred_data, axis=0)
        high = np.max(pred_data, axis=0)
        delta = high - low + 1e-15
        normalized_data = (pred_data - low) / delta
        return normalized_data, low, high

    def data_denormalize(self, normalized_data, low, high):
        delta = high - low + 1e-15
        preds = normalized_data * delta + low
        return preds


if __name__ == '__main__':
    df = pd.read_csv('climate_ts_data.csv', index_col=['time'])
    pred_data = df.iloc[:, :1].values
    aux_info = df.iloc[:, 1:].values
    nums = pred_data.shape[0]
    features = pred_data.shape[-1]

    transformer = TransformerPredictor()
    preds = transformer.predict(pred_data, nums, features, aux_info=aux_info, pred_preiod=10, epochs=5)
    print(preds)


