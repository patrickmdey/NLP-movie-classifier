import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score
from loguru import logger
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as mtick


def percentage_heatmap_from_dataframe(dataframe, title, path):
    plt.clf()
    cmap = sns.color_palette("light:r", as_cmap=True, n_colors=5)

    # Create a figure and axis for the heatmap
    fig, ax = plt.subplots()

    # Normalize the values in the DataFrame
    # dataframe = dataframe / dataframe.sum().sum()

    # Plot the heatmap
    heatmap = sns.heatmap(dataframe, annot=True, fmt=".1%",
                          linewidths=0.5, ax=ax, cmap=cmap, cbar=True)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=1.0, decimals=0))

    # Set axis labels and title
    ax.set_xticklabels(dataframe.columns, rotation=0, ha='right', fontsize=10)
    ax.set_yticklabels(dataframe.index, rotation=0, fontsize=10)
    ax.set_title(title)

    # Show the plot
    plt.tight_layout()
    path = path + title.replace(" ", "_").lower()
    plt.savefig(path + ".png")


def get_confusion_matrix(df):
    true_labels = df['label'].values
    predicted_labels = df['prediction'].values
    return multilabel_confusion_matrix(true_labels, predicted_labels)


def get_accuracy(df):
    true_labels = df['label'].values
    predicted_labels = df['prediction'].values
    return accuracy_score(true_labels, predicted_labels)


def get_precision(df):
    true_labels = df['label'].values
    predicted_labels = df['prediction'].values
    return precision_score(true_labels, predicted_labels, average='macro')


def plot_accuracy_per_epoch(df, model):
    plt.clf()
    df = df[["epoch", "prediction", 'label']]
    df = df[df["epoch"] != "epoch"]
    df = df.astype({"epoch": int, "prediction": int, "label": int})
    accuracy_per_epoch = df.groupby("epoch").apply(
        lambda x: accuracy_score(x["label"], x["prediction"]))
    accuracy_per_epoch.plot(color="red")
    plt.title(f"Accuracy per epoch - {model}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"out/{model}/post_process/accuracy.png")


def plot_precision_per_epoch(df, model):
    plt.clf()
    df = df[["epoch", "prediction", 'label']]
    df = df[df["epoch"] != "epoch"]
    df = df.astype({"epoch": int, "prediction": int, "label": int})

    # Choose an appropriate average setting
    precision_per_epoch = df.groupby("epoch").apply(
        lambda x: precision_score(x["label"], x["prediction"], average='weighted'))

    precision_per_epoch.plot(color="red")
    plt.title(f"Precision per epoch - {model}")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.savefig(f"out/{model}/post_process/precision.png")


def calculate_results(df, confusion_matrix=None):
    correct = 0
    for _, row in df.iterrows():
        pred = row['prediction']
        rating = row['label']
        if pred == rating:
            correct += 1

        if confusion_matrix is not None:
            confusion_matrix[rating][pred] += 1

    return correct


def solo_metrics(df, path):
    confusion_matrixes = get_confusion_matrix(df)
    confusion_dict = {
        rating: {pred: 0 for pred in range(1, 6)} for rating in range(1, 6)}

    def confusion_row_to_percent(row):
        total = row.sum()
        return row.apply(lambda x: (x / total).round(4))

    calculate_results(df[["prediction", 'label']], confusion_dict)

    confusion_df = pd.DataFrame(confusion_dict)
    confusion_df = confusion_df.apply(confusion_row_to_percent, axis=1)

    percentage_heatmap_from_dataframe(
        confusion_df, "Confusion matrix for all reviews", path)

    for i, cm in enumerate(confusion_matrixes):
        rating = i+1
        rating_df = df[df['label'] == rating]
        percentage_heatmap_from_dataframe(pd.DataFrame(
            cm), f"Confusion matrix for reviews of rating {rating}", path)
        logger.info(f"Accuracy for rating {rating}: {get_accuracy(rating_df)}")
        logger.info(
            f"Precision for rating {rating}: {get_precision(rating_df)}")

    accuracy = get_accuracy(df)
    precision = get_precision(df)
    logger.info(f"Accuracy for all reviews: {accuracy}")
    logger.info(f"Precision for all reviews: {precision}")


if __name__ == "__main__":
    logger.info("Starting epoch analysis")
    model = "roberta"
    dataset_path = f"out/{model}/progress/classify.csv"
    data = pd.read_csv(dataset_path, sep=";")
    data = data.dropna()

    path = f"out/{model}/post_process/"
    if not os.path.exists(path):
        os.makedirs(path)

    solo_metrics(data, path)
    plot_precision_per_epoch(data, model)
    plot_accuracy_per_epoch(data, model)
