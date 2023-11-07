import matplotlib.pyplot as plt
import pandas as pd
import json
from utils import reduce_columns
import os
import re
from loguru import logger

def plot_rating_distribution(df, dataset_name, out_path):
    logger.info("Plotting rating distribution")
    plt.clf()
    value_counts = df["rating"].value_counts()
    labels = value_counts.index
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    plt.pie(value_counts,
            labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title(f"Rating distribution - {dataset_name}")
    plt.tight_layout()
    plt.savefig(out_path + "/plots/rating_distribution.png")
    plt.close()

def plt_avg_word_per_rating(df, dataset_name, out_path, write_to_file=True):
    logger.info("Plotting average word count per rating")
    plt.clf()
    df["word_count"] = df["review"].apply(lambda x: len(str(x).split(" ")))
    avg_word_count = df.groupby("rating")["word_count"].mean()
    plt.bar(avg_word_count.index, avg_word_count.values)
    plt.title(f"Average word count per rating - {dataset_name}")
    plt.xlabel("Rating")
    plt.ylabel("Average word count")
    plt.xticks(avg_word_count.index)
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path + "/plots/avg_word_count_per_rating.png")
    plt.close()
    if write_to_file:
        avg_word_count.to_csv(out_path + "/avg_word_count_per_rating.csv")

def get_most_used_words_per_rating(df, out_path, word_amount=50):
    ratings = df["rating"].unique()

    for rating in ratings:
        logger.info(f"Getting most used words for rating {rating}")
        words = {}
        for review in df[df["rating"] == rating]["review"]:
            for word in review.split(" "):
                word = word.strip()
                word = re.sub(r"[^a-zA-Z0-9]+", '', word.lower())
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}
        word_df = pd.DataFrame.from_dict(words, orient="index", columns=["count"])
        word_df.rename(columns={"index": "word"}, inplace=True)
        word_df = word_df.head(word_amount)
        word_df.to_csv(f"{out_path}/most_used_words_for_{rating}_rating.csv")
    


if __name__ == "__main__":
    """
    This script analyses the dataset according to the config_analysis.json file.
    Example:
    {
        "dataset": "stanford.csv",
        "csv_sep": ",",
        "rating_col": "Rating",
        "review_col": "Review"
    }
    """

    config = json.load(open("config_analysis.json"))

    dataset_path = config["dataset"]
    csv_sep = config["csv_sep"] if "csv_sep" in config else ";"
    rating_col = config["rating_col"] if "rating_col" in config else "rating"
    review_col = config["review_col"] if "review_col" in config else "review"

    dataset_name = dataset_path.split("/")[-1].split(".")[0] #CHECK: This is not optimal

    df = pd.read_csv(dataset_path, sep=csv_sep, on_bad_lines='skip')
    df = df.dropna()
    df = reduce_columns(df, rating_col, review_col)

    out_path = f"out/analysis/{dataset_name}"
    if not os.path.exists(out_path + "/plots"):
        os.makedirs(out_path + "/plots")

    plot_rating_distribution(df, dataset_name, out_path)
    plt_avg_word_per_rating(df, dataset_name, out_path)
    get_most_used_words_per_rating(df, out_path)





