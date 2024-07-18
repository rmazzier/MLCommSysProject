import time
import os
import torch
import wandb
from torch import nn, optim
import matplotlib.pyplot as plt
from tqdm import tqdm


from dataset import Rastro_Dataset
from constants import DEVICE
from utils import plot_forecast_example, SPLIT
from models import Seq2SeqRNN


def train_epoch(train_dataloader, valid_dataloader, net, optimizer, criterion, teacher_forcing=False, max_iters=None):

    total_loss_train = 0
    total_loss_valid = 0

    # Training
    net.train()

    for i, data in tqdm(enumerate(train_dataloader)):
        if max_iters is not None and i >= max_iters:
            break
        input_tensor, target_tensor = data
        input_tensor = input_tensor.float().to(DEVICE)
        target_tensor = target_tensor.float().to(DEVICE)

        optimizer.zero_grad()

        # encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)

        # # To use teacher forcing, we feed the target tensor as well
        # decoder_outputs, _, _ = decoder(
        #     encoder_outputs, encoder_hidden, encoder_cell)

        decoder_outputs, _ = net(
            input_tensor, target_tensor if teacher_forcing else None)

        loss = criterion(
            decoder_outputs,
            target_tensor
        )
        loss.backward()
        if i % 50 == 0:
            print(f"Loss: {loss.item()}")

        optimizer.step()

        total_loss_train += loss.item()

    # Validation
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_dataloader)):
            input_tensor, target_tensor = data
            input_tensor = input_tensor.float().to(DEVICE)
            target_tensor = target_tensor.float().to(DEVICE)

            # No Teacher forcing
            decoder_outputs, _ = net(input_tensor, None)

            loss = criterion(
                decoder_outputs,
                target_tensor
            )

            total_loss_valid += loss.item()

    total_loss_train = total_loss_train / len(train_dataloader)
    total_loss_valid = total_loss_valid / len(valid_dataloader)

    return total_loss_train, total_loss_valid


def train(config, train_dataloader, valid_dataloader, test_dataloader, net, learning_rate=0.001,
          print_every=1, plot_every=100):

    wandb.login()
    run = wandb.init(
        project="MLComSysProject",
        config=config,
        name=config["MODEL_NAME"],
        notes=config["NOTES"],
        reinit=True,
        mode=config["WANDB_MODE"],
    )

    wandb.watch(net)

    # start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=0.0001)
    # criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    for epoch in range(1, config["EPOCHS"] + 1):
        print(f"Start epoch {epoch}")
        train_loss, valid_loss = train_epoch(train_dataloader,
                                             valid_dataloader,
                                             net,
                                             optimizer,
                                             criterion,
                                             teacher_forcing=config["TEACHER_FORCING"])

        wandb.log({"Train Loss": train_loss,
                   "Validation Loss": valid_loss}, step=epoch)

    # Test phase
    test_step(test_dataloader, net, criterion)

    run.finish()
    return encoder, decoder


def test_step(test_dataloader, net, criterion):
    net.eval()

    test_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader)):
            input_tensor, target_tensor = data
            input_tensor = input_tensor.float().to(DEVICE)
            target_tensor = target_tensor.float().to(DEVICE)

            decoder_outputs, _ = net(
                input_tensor, None)

            loss = criterion(
                decoder_outputs,
                target_tensor
            )

            test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {test_loss}")
    wandb.log({"Test Loss": test_loss})


def setup_training(config, agent_idx=-1):
    train_dataset = Rastro_Dataset(
        config=config,
        agent_idx=agent_idx,
        split=SPLIT.TRAIN)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        drop_last=False,
        num_workers=os.cpu_count())

    valid_dataset = Rastro_Dataset(
        config=config,
        agent_idx=agent_idx,
        split=SPLIT.VALIDATION)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count())

    test_dataset = Rastro_Dataset(
        config=config,
        agent_idx=agent_idx,
        split=SPLIT.TEST)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        drop_last=False,
        num_workers=os.cpu_count())

    # Get the input size by sampling a single sample
    input_size = next(iter(train_loader))[0][0].shape[-1]

    # Get the output size by sampling a single label
    output_size = next(iter(train_loader))[1][0].shape[-1]

    net = Seq2SeqRNN(config, input_size, output_size,
                     use_attention=False).to(DEVICE)

    return train_loader, valid_loader, test_dataset, test_loader, net


if __name__ == "__main__":
    from constants import CONFIG
    import json

    Rastro_Dataset.generate_data(
        config=CONFIG, split_seed=123, standardize=True)

    # Rastro_Dataset.generate_data_simple(
    #     config=CONFIG, split_seed=123, standardize=True
    # )

    train_loader, valid_loader, test_dataset, test_loader, net = setup_training(
        CONFIG, agent_idx=-1)

    encoder, decoder = train(config=CONFIG,
                             train_dataloader=train_loader,
                             valid_dataloader=valid_loader,
                             test_dataloader=test_loader,
                             net=net,
                             learning_rate=0.001)

    # Save logs and weights the trained models
    run_results_dir = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["MODEL_NAME"])
    os.makedirs(run_results_dir, exist_ok=True)
    torch.save(encoder, os.path.join(run_results_dir, "encoder.pt"))
    torch.save(decoder, os.path.join(run_results_dir, "decoder.pt"))

    # Also save a copy of the relative config file
    with open(os.path.join(run_results_dir, "config.json"), 'w') as f:
        json.dump(CONFIG, f)

    # Plot an example of forecasting from the test set
    # plot_forecast_example(test_dataset, encoder, decoder)
