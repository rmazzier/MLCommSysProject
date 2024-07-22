import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE


# Default tokens to denote the start and end of a sentence
SOS_token = 0


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()

        # self.rec_layer = nn.GRU(input_size=input_size,
        #                         hidden_size=config["HIDDEN_SIZE"],
        #                         num_layers=config["NUM_LAYERS"],
        #                         batch_first=True)
        self.cell_type = config["CELL_TYPE"]

        # LSTM layer
        if self.cell_type == "LSTM":
            self.rec_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=config["HIDDEN_SIZE"],
                num_layers=config["NUM_LAYERS"],
                bias=True,
                batch_first=True,
                dropout=dropout_p,
                bidirectional=config["BIDIRECTIONAL"]
            )
        elif self.cell_type == "GRU":
            self.rec_layer = nn.GRU(
                input_size=input_size,
                hidden_size=config["HIDDEN_SIZE"],
                num_layers=config["NUM_LAYERS"],
                bias=True,
                batch_first=True,
                dropout=dropout_p,
                bidirectional=config["BIDIRECTIONAL"]
            )
        else:
            raise ValueError("Invalid cell type")

    def forward(self, input):
        if self.cell_type == "LSTM":
            output, (hidden, cell) = self.rec_layer(input)
            return output, hidden, cell
        else:
            output, hidden = self.rec_layer(input)
            return output, hidden


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()

        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):

        # Additive Attention
        # Take only the last hidden state of the encoder
        query = query[:, -1, :].unsqueeze(1)
        # scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))

        # Dot product between the query and the keys
        scores = torch.bmm(self.Ua(keys), self.Wa(query).permute(0, 2, 1))

        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, config, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = config["HIDDEN_SIZE"]
        self.config = config
        self.use_attention = config["ATTENTION"]

        if config["BIDIRECTIONAL"]:
            self.D = 2
        else:
            self.D = 1

        if self.use_attention:

            self.attention = AttentionLayer(self.hidden_size * self.D)
            input_size = self.hidden_size * self.D + output_size

        else:
            input_size = output_size

        self.cell_type = config["CELL_TYPE"]
        if self.cell_type == "LSTM":
            self.rec_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=config["NUM_LAYERS"],
                bias=True,
                batch_first=True,
                dropout=dropout_p,
                bidirectional=config["BIDIRECTIONAL"]
            )
        elif self.cell_type == "GRU":
            self.rec_layer = nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=config["NUM_LAYERS"],
                bias=True,
                batch_first=True,
                dropout=dropout_p,
                bidirectional=config["BIDIRECTIONAL"]
            )
        else:
            raise ValueError("Invalid cell type")

        self.out = nn.Linear(self.hidden_size * self.D, output_size)

    def forward(self, encoder_outputs, encoder_hidden, encoder_cell, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        # decoder_input = torch.empty(
        #     batch_size, 1, self.hidden_size, dtype=torch.float, device=DEVICE).fill_(SOS_token)
        decoder_input = torch.empty(
            batch_size, 1, 3, dtype=torch.float, device=DEVICE).fill_(SOS_token)

        # Encoder hidden has shape (num_layers, batch_size, hidden_size)
        # We just want the hidden state of the last layer

        decoder_hidden = encoder_hidden

        if self.cell_type == "LSTM":
            # decoder_cell = encoder_cell[-1].unsqueeze(0)
            decoder_cell = encoder_cell
        elif self.cell_type == "GRU":
            decoder_cell = None

        predictions = []
        attentions = []

        for i in range(self.config["N_TO_PREDICT"]):
            prediction, decoder_output, decoder_hidden, decoder_cell, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs
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
        # predictions = F.log_softmax(predictions, dim=-1)
        if attentions[0] == None:
            attentions = None
        else:
            attentions = torch.cat(attentions, dim=1)

        return predictions, decoder_hidden, attentions

    def forward_step(self, decoder_input, decoder_hidden, decoder_cell, encoder_outputs):
        if self.use_attention:

            if self.config["BIDIRECTIONAL"]:
                # Concatenate the hidden states of the forward and backward passes
                query = torch.cat(
                    (decoder_hidden[0], decoder_hidden[1]), axis=1).unsqueeze(0)

                query = query.permute(1, 0, 2)
            else:
                query = decoder_hidden.permute(1, 0, 2)

            context, attn_weights = self.attention(query, encoder_outputs)
            rec_layer_input = torch.cat((decoder_input, context), dim=2)
        else:
            rec_layer_input = decoder_input
            attn_weights = None

        if self.cell_type == "LSTM":
            decoder_output, (decoder_hidden, decoder_cell) = self.rec_layer(
                rec_layer_input, (decoder_hidden, decoder_cell))
        elif self.cell_type == "GRU":
            decoder_output, decoder_hidden = self.rec_layer(
                rec_layer_input, decoder_hidden)
            decoder_cell = None

        prediction = self.out(decoder_output)
        return prediction, decoder_output, decoder_hidden, decoder_cell, attn_weights


class Seq2SeqRNN(nn.Module):
    def __init__(self, config, input_size, output_size):
        super(Seq2SeqRNN, self).__init__()
        self.config = config

        self.encoder = EncoderRNN(self.config, input_size)

        self.decoder = AttnDecoderRNN(self.config, output_size)

    def forward(self, input_tensor, target_tensor=None):

        if self.config["CELL_TYPE"] == "LSTM":
            encoder_outputs, encoder_hidden, encoder_cell = self.encoder(
                input_tensor)
        elif self.config["CELL_TYPE"] == "GRU":
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            encoder_cell = None
        else:
            raise ValueError("Invalid cell type")

        # To use teacher forcing, we have to feed the target tensor as well
        # in the input of the decoder
        # if not, defaults to none
        predictions, _, _ = self.decoder(
            encoder_outputs, encoder_hidden, encoder_cell, target_tensor)

        return predictions
