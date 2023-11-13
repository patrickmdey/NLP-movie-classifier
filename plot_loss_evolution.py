

import json
import matplotlib.pyplot as plt
import os
from loguru import logger

if __name__ == "__main__":


    # PATH = "out/roberta/imdb80-checkpoints/2023_11_12_18_09_23"
    PATH = "out/bert/imdb80-checkpoints/20231112_153553"

    trainer_state_files = []

    train_state_file_names = os.listdir(PATH)
    train_state_file_names = sorted(
        train_state_file_names, key=lambda x: int(x.split('-')[1]))

    for file_name in train_state_file_names:
        files = os.listdir(f"{PATH}/{file_name}")
        for file in files:
            if file.startswith("trainer_state"):
                trainer_state_files.append(f"{file_name}/{file}")
    # train_state_file_names = [
    #     file_name for file_name in train_state_file_names if file_name.startswith("train_state")]

    checkpoint_losses = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}

    for i, file_name in enumerate(trainer_state_files):
        data = json.load(open(f"{PATH}/{file_name}"))

        # Extract values from log_history
        log_history = data["log_history"]
        epoch_loss_data = []
        epoch_eval_loss = []
        for entry in log_history[:-1]:
            if "loss" not in entry:
                eval_loss = entry["eval_loss"]
                eval_epoch = entry["epoch"]
                epoch_eval_loss.append(

                    {"epoch": eval_epoch, "loss": eval_loss})
            else:
                loss = entry["loss"]
                epoch = entry["epoch"]
                epoch_loss_data.append({"epoch": epoch, "loss": loss})

        checkpoint_losses[i]["epoch_loss_data"] = epoch_loss_data.copy()
        checkpoint_losses[i]["epoch_eval_loss"] = epoch_eval_loss.copy()

    # plot loss evolution over epochs
    for checkpoint in checkpoint_losses:

        path = f"out/roberta/loss_plots/imdb80-checkpoints/{checkpoint+1}"
        if not os.path.exists(path):
            os.makedirs(path)

        plt.clf()
        x = [entry["epoch"]
             for entry in checkpoint_losses[checkpoint]["epoch_loss_data"]]
        y = [entry["loss"]
             for entry in checkpoint_losses[checkpoint]["epoch_loss_data"]]
        plt.plot(x, y, color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss evolution over epochs - {checkpoint+1}")
        plt.tight_layout()
        plt.savefig(f"{path}/loss_evolution.png")

        # plot eval loss evolution over epochs
        plt.clf()
        x = [entry["epoch"]
             for entry in checkpoint_losses[checkpoint]["epoch_eval_loss"]]
        y = [entry["loss"]
             for entry in checkpoint_losses[checkpoint]["epoch_eval_loss"]]
        plt.plot(x, y, color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Eval Loss")
        plt.title(f"Eval loss evolution over epochs - {checkpoint+1}")
        plt.tight_layout()
        plt.savefig(f"{path}/eval_loss_evolution.png")

    # plot all epoch_loss_data curves on same plot
    plt.clf()
    for checkpoint in checkpoint_losses:
        logger.info(f"{checkpoint_losses[checkpoint]['epoch_loss_data']}")
        plt.plot([entry["epoch"] for entry in checkpoint_losses[checkpoint]["epoch_loss_data"]], [
            entry["loss"] for entry in checkpoint_losses[checkpoint]["epoch_loss_data"]], label=f"Checkpoint {checkpoint+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss evolution over epochs - All Checkpoints")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "out/roberta/loss_plots/imdb80-checkpoints/loss_evolution_all_checkpoints.png")
