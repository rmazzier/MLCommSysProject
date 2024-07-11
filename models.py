import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE


# Define the input and output sequence lengths
input_seq_len = 10
output_seq_len = 5

# Define the number of features
num_features = 8

# Default tokens to denote the start and end of a sentence
SOS_token = 0
EOS_token = 1


# Define the model
class EncoderRNN(nn.Module):
    def __init__(self, config, input_size, hidden_size, num_layers=1, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.config = config
        self.gru = nn.GRU(output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, self.output_size, dtype=torch.float, device=DEVICE).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.config["N_TO_PREDICT"]):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i, :].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                # detach from history as input
                decoder_input = decoder_output.detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        # We return `None` for consistency in the training loop
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        # output = self.embedding(input)
        # output = F.relu(input)
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=DEVICE).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.config["N_TO_PREDICT"]):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(
                    1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                # detach from history as input
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
