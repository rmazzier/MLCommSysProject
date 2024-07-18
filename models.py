import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE


# Default tokens to denote the start and end of a sentence
SOS_token = 0


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()

        self.rec_layer = nn.GRU(input_size=input_size,
                                hidden_size=config["HIDDEN_SIZE"],
                                num_layers=config["NUM_LAYERS"],
                                batch_first=True)

        # self.conv1d = nn.Conv1d(
        #     in_channels=input_size,
        #     out_channels=config["HIDDEN_SIZE"],
        #     kernel_size=config["KERNEL_SIZE"],
        #     stride=1,
        #     padding=0,
        #     groups=1,
        #     bias=True,
        #     padding_mode='zeros'
        # )

        # self.rec_layer = nn.LSTM(
        #     input_size=input_size,
        #     hidden_size=config["HIDDEN_SIZE"],
        #     num_layers=num_layers,
        #     bias=True,
        #     batch_first=True,
        #     dropout=dropout_p,
        #     bidirectional=False
        # )

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        x = self.dropout(input)
        output, hidden = self.rec_layer(x)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config, output_size):
        super(DecoderRNN, self).__init__()
        self.config = config
        self.hidden_size = config["HIDDEN_SIZE"]
        self.rec_layer = nn.GRU(
            input_size=output_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, output_size)

        self.output_size = output_size

    def forward(self, encoder_outputs, encoder_hidden, encoder_cell, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, self.output_size, dtype=torch.float, device=DEVICE).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        decoder_outputs = []

        for i in range(self.config["N_TO_PREDICT"]):
            decoder_output, decoder_hidden, decoder_cell = self.forward_step(
                decoder_input, decoder_hidden, decoder_cell)
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

    def forward_step(self, input, hidden, cell):
        # output = self.embedding(input)
        # output = F.relu(input)
        output, hidden = self.rec_layer(
            input, hidden[-1].unsqueeze(0))
        output = self.out(output)
        return output, hidden, cell


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):

        # Additive Attention
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))

        # Dot product between the query and the keys
        # scores = torch.bmm(self.Ua(keys), self.Wa(query).permute(0, 2, 1))

        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, config, output_size, dropout_p=0.1, use_attention=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = config["HIDDEN_SIZE"]
        self.config = config
        self.attention = AttentionLayer(self.hidden_size)

        self.use_attention = use_attention

        if use_attention:
            self.rec_layer = nn.GRU(self.hidden_size + output_size,
                                    self.hidden_size, batch_first=True)
        else:
            self.rec_layer = nn.GRU(output_size,
                                    self.hidden_size, batch_first=True)

        self.out = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        # decoder_input = torch.empty(
        #     batch_size, 1, self.hidden_size, dtype=torch.float, device=DEVICE).fill_(SOS_token)
        decoder_input = torch.empty(
            batch_size, 1, 3, dtype=torch.float, device=DEVICE).fill_(SOS_token)

        # Encoder hidden has shape (num_layers, batch_size, hidden_size)
        # We just want the hidden state of the last layer
        decoder_hidden = encoder_hidden[-1].unsqueeze(0)
        predictions = []
        attentions = []

        for i in range(self.config["N_TO_PREDICT"]):
            prediction, decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            predictions.append(prediction)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i, :].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                # detach from history as input

                # decoder_input = decoder_output.detach()
                decoder_input = prediction.detach()

        predictions = torch.cat(predictions, dim=1)
        predictions = F.log_softmax(predictions, dim=-1)
        if attentions[0] == None:
            attentions = None
        else:
            attentions = torch.cat(attentions, dim=1)

        return predictions, decoder_hidden, attentions

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        # embedded = self.dropout(self.embedding(decoder_input))
        if self.use_attention:

            query = decoder_hidden.permute(1, 0, 2)
            context, attn_weights = self.attention(query, encoder_outputs)
            input_gru = torch.cat((decoder_input, context), dim=2)
        else:
            input_gru = decoder_input
            attn_weights = None

        decoder_output, decoder_hidden = self.rec_layer(
            input_gru, decoder_hidden)
        prediction = self.out(decoder_output)

        return prediction, decoder_output, decoder_hidden, attn_weights


class Seq2SeqRNN(nn.Module):
    def __init__(self, config, input_size, output_size, use_attention=True):
        super(Seq2SeqRNN, self).__init__()
        self.encoder = EncoderRNN(
            config, input_size)
        self.decoder = AttnDecoderRNN(
            config, output_size, dropout_p=0.1, use_attention=use_attention)
        self.teacher_forcing = config["TEACHER_FORCING"]

    def forward(self, input_tensor, target_tensor=None):
        encoder_outputs, encoder_hidden = self.encoder(
            input_tensor)

        if self.teacher_forcing:
            decoder_outputs, decoder_hidden, _ = self.decoder(
                encoder_outputs, encoder_hidden, target_tensor)
        else:
            decoder_outputs, decoder_hidden, _ = self.decoder(
                encoder_outputs, encoder_hidden)

        return decoder_outputs, decoder_hidden
