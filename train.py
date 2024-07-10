import time
import torch
from torch import nn, optim

import matplotlib.pyplot as plt

from tqdm import tqdm

from constants import DEVICE


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):

    total_loss = 0
    for i, data in tqdm(enumerate(dataloader)):
        input_tensor, target_tensor = data
        input_tensor = input_tensor.float().to(DEVICE)
        target_tensor = target_tensor.float().to(DEVICE)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(
            encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs,
            target_tensor
        )
        loss.backward()
        if i % 50 == 0:
            print(f"MSE Loss: {loss.item()}")

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    # criterion = nn.NLLLoss()
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        print(f"Start epoch {epoch}")
        loss = train_epoch(train_dataloader, encoder, decoder,
                           encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
            #                              epoch, epoch / n_epochs * 100, print_loss_avg))
            print(f"Epoch: {epoch}, Loss: {print_loss_avg}")

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # plot losses
    plt.figure()
    plt.plot(plot_losses)
    plt.show()


if __name__ == "__main__":
    from constants import CONFIG
    from dataset import Rastro_Dataset
    from utils import SPLIT

    from models import EncoderRNN, DecoderRNN

    train_dataset = Rastro_Dataset(
        config=CONFIG, agent_idx=-1, split=SPLIT.TRAIN)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

    # Get the input size by sampling a single sample
    input_size = next(iter(train_dataloader))[0][0].shape[-1]

    # Get the output size by sampling a single label
    output_size = next(iter(train_dataloader))[1][0].shape[-1]

    encoder = EncoderRNN(config=CONFIG, input_size=input_size,
                         hidden_size=256).to(DEVICE)
    decoder = DecoderRNN(config=CONFIG, hidden_size=256,
                         output_size=output_size).to(DEVICE)

    train(train_dataloader=train_dataloader,
          encoder=encoder, decoder=decoder, n_epochs=CONFIG["EPOCHS"])
