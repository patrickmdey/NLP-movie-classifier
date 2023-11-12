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
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)
    
    # Create a figure and axis for the heatmap
    fig, ax = plt.subplots()

    # Normalize the values in the DataFrame
    # dataframe = dataframe / dataframe.sum().sum()
    
    # Plot the heatmap
    heatmap = sns.heatmap(dataframe, annot=True, fmt=".1%", linewidths=0.5, ax=ax, cmap=cmap, cbar=True)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

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

def plot_accuracy_per_epoch(df):
    epochs = df['epoch'].unique()

    confusion_matrices = []
    accuracies = []

    logger.info("Calculating confusion matrices and accuracies")
    for epoch in epochs:
        # Filter the df for the current epoch
        epoch_df = df[df['epoch'] == epoch]
        confusion_matrices.append(get_confusion_matrix(epoch_df))
        # Calculate accuracy for the current epoch
        accuracies.append(get_accuracy(epoch_df))

    logger.info(f"Accuracy per epoch: {accuracies}")

    logger.info("Saving confusion matrices")
    # Plot the accuracy for each epoch
    plt.plot(epochs, accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    
    plt.savefig('accuracy.png')

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
    confusion_dict = {rating: {pred: 0 for pred in range(1, 6)} for rating in range(1, 6)}
    def confusion_row_to_percent(row):
        total = row.sum()
        return row.apply(lambda x: (x / total).round(4))

    df = df[["epoch", "prediction", 'label']]
    df = df[df["epoch"] == "5"]

    
    
    calculate_results(df, confusion_dict)

    
    confusion_df = pd.DataFrame(confusion_dict)
    confusion_df = confusion_df.apply(confusion_row_to_percent, axis=1)

    percentage_heatmap_from_dataframe(confusion_df, "Confusion matrix for all reviews", path)

    for i, cm in enumerate(confusion_matrixes):
        rating = i+1
        rating_df = df[df['label'] == rating]
        percentage_heatmap_from_dataframe(pd.DataFrame(cm), f"Confusion matrix for reviews of rating {rating}", path)
        logger.info(f"Accuracy for rating {rating}: {get_accuracy(rating_df)}")
        logger.info(f"Precision for rating {rating}: {get_precision(rating_df)}")
        
    accuracy = get_accuracy(df)
    precision = get_precision(df)
    logger.info(f"Accuracy for all reviews: {accuracy}")
    logger.info(f"Precision for all reviews: {precision}")

if __name__ == "__main__":
    logger.info("Starting epoch analysis")
    dataset_path = "out/bert/progress/classify.csv"
    data = pd.read_csv(dataset_path, sep=";")
    data = data.dropna()

    path = "out/bert/post_process/"
    if not os.path.exists(path):
        os.makedirs(path)

    # plot_accuracy_per_epoch(data)
    solo_metrics(data, path)