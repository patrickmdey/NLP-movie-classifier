import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score

df = pd.read_csv("out/bert/progress/classify.csv", sep=";", encoding="utf-8")

# epoch;text;label;prediction

def plot_accuracy_per_epoch(df):
    plt.clf()
    df = df[["epoch", "prediction", 'label']]
    df = df[df["epoch"] != "epoch"]
    df = df.astype({"epoch": int, "prediction": int, "label": int})
    accuracy_per_epoch = df.groupby("epoch").apply(lambda x: accuracy_score(x["label"], x["prediction"]))
    accuracy_per_epoch.plot()
    plt.title("Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig('accuracy.png')

def plot_precision_per_epoch(df):
    plt.clf()
    df = df[["epoch", "prediction", 'label']]
    df = df[df["epoch"] != "epoch"]
    df = df.astype({"epoch": int, "prediction": int, "label": int})
    
    # Choose an appropriate average setting
    precision_per_epoch = df.groupby("epoch").apply(lambda x: precision_score(x["label"], x["prediction"], average='weighted'))
    
    precision_per_epoch.plot()
    plt.title("Precision per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.savefig('precision.png')

if __name__ == "__main__":
    plot_accuracy_per_epoch(df)
    plot_precision_per_epoch(df)