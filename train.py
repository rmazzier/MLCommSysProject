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
from models import Seq2SeqRNN, EncoderRNN, AttnDecoderRNN


def train_epoch(
        config,
        train_dataloader,
        valid_dataloader,
        net,
        optimizer,
        criterion,
        max_iters=None):

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

        predictions = net(
            input_tensor, target_tensor if config["TEACHER_FORCING"] else None)

        loss = criterion(
            predictions,
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

            # No Teacher forcing since we are in validation mode
            predictions = net(input_tensor)

            loss = criterion(
                predictions,
                target_tensor
            )

            total_loss_valid += loss.item()

    total_loss_train = total_loss_train / len(train_dataloader)
    total_loss_valid = total_loss_valid / len(valid_dataloader)

    return total_loss_train, total_loss_valid


def train(config, train_dataloader, valid_dataloader, test_dataloader, net, learning_rate=0.001):

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

    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=0.001)

    if config["CRITERION"] == "MSE":
        criterion = nn.MSELoss()
    elif config["CRITERION"] == "MAE":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Invalid criterion")

    best_valid_loss = float("inf")
    run_results_dir = os.path.join(config["RESULTS_DIR"], config["MODEL_NAME"])
    os.makedirs(run_results_dir, exist_ok=True)

    for epoch in range(1, config["EPOCHS"] + 1):
        print(f"Start epoch {epoch}")
        train_loss, valid_loss = train_epoch(config,
                                             train_dataloader,
                                             valid_dataloader,
                                             net,
                                             optimizer,
                                             criterion)

        wandb.log({"Train Loss": train_loss,
                   "Validation Loss": valid_loss}, step=epoch)

        if valid_loss < best_valid_loss and config["WANDB_MODE"] == "online":
            best_valid_loss = valid_loss
            # Save the model
            torch.save(net.state_dict(), os.path.join(
                run_results_dir, "model_weights.pt"))

            # Save online
            wandb.save(os.path.join(run_results_dir, "model_weights.pt"))

    # Test phase
    test_step(config, test_dataloader, net, criterion)

    run.finish()
    return encoder, decoder


def test_step(config, test_dataloader, net, criterion):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader)):
            input_tensor, target_tensor = data
            input_tensor = input_tensor.float().to(DEVICE)
            target_tensor = target_tensor.float().to(DEVICE)

            # No Teacher forcing since we are in validation mode
            predictions = net(input_tensor)

            loss = criterion(
                predictions,
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

    net = Seq2SeqRNN(config, input_size, output_size).to(DEVICE)

    # encoder = EncoderRNN(config, input_size=input_size,
    #                      dropout_p=0.1).to(DEVICE)
    # decoder = AttnDecoderRNN(config, output_size, dropout_p=0.1).to(DEVICE)
    # return train_loader, valid_loader, test_dataset, test_loader, encoder, decoder
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

    for j in range(10):
        plot_forecast_example(CONFIG, test_dataset, net, to_plot_idx=j)

    # Save logs and weights the trained models
    run_results_dir = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["MODEL_NAME"])
    os.makedirs(run_results_dir, exist_ok=True)

    # Also save a copy of the relative config file
    with open(os.path.join(run_results_dir, "config.json"), 'w') as f:
        json.dump(CONFIG, f)

    # Plot an example of forecasting from the test set
    # plot_forecast_example(test_dataset, encoder, decoder)
