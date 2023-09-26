import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

def plot_score_distribution(df):
    plt.clf()
    score_counts = df['score'].value_counts().sort_index()
    plt.pie(score_counts,
            labels=score_counts.index, autopct='%1.1f%%',)
    plt.title("Score distribution")
    out_path = "out/score_distribution"
    plt.tight_layout()
    plt.savefig(out_path + ".png")
    plt.close()


def plt_avg_words_per_rating():
    plt.clf()
    df = pd.read_csv("out/avg_words_per_rating.csv")
    plt.bar(df["rating"], df["avg_words"])
    plt.xlabel("Rating")
    plt.ylabel("Average amount of words")
    plt.title("Average amount of words per rating after removing most used words")
    plt.tight_layout()
    plt.savefig("out/avg_words_per_rating.png")
    plt.close()


def get_most_used_words(df):
    words = {}
    avg_amount_of_words = 0
    for review in df["review_text"]:
        avg_amount_of_words += len(review.split())
        for word in review.split():
            if word not in words:
                words[word] = 0
            words[word] += 1
    words = {k: v for k, v in sorted(
        words.items(), key=lambda item: item[1], reverse=True)}
    return words, avg_amount_of_words/len(df)


def write_most_used_words(df, amount=30):
    most_used_words, _ = get_most_used_words(df)
    with open("out/most_used_words.csv", "w") as f:
        f.write("word,count\n")
        for word, count in list(most_used_words.items())[:amount]:
            f.write(word + "," + str(count) + "\n")
    f.close()

    return list(most_used_words.keys())[:amount]

def write_most_used_words_per_rating(df, ratings, amount=30):
    most_used_words_with = {}

    with open("out/avg_words_per_rating.csv", "w") as avg_word_f:
        avg_word_f.write("rating,avg_words\n")
        for rating in ratings:
            rating_df = df[df["score"] == rating]
            most_used_words_with[rating], avg_amount = get_most_used_words(
                rating_df)
            avg_word_f.write(str(rating) + "," + str(avg_amount) + "\n")

            with open("out/most_used_words_" + str(rating)+".csv", "w") as f:
                f.write("word,count\n")
                for word, count in list(most_used_words_with[rating].items())[:amount]:
                    f.write(word + "," + str(count) + "\n")
            f.close()
    avg_word_f.close()
    plt_avg_words_per_rating()


def preprocess_dataset(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df["score"] = df["score"].astype(int)
    return df


def remove_stopwords(text, stop_words):
    return " ".join([word for word in text.split() if word not in stop_words])

if __name__ == "__main__":
    stop_words = [
    'the', 'and', 'of', 'a', 'in', 'to', 'it', 'is', 'was', 'I', 
    'for', 'on', 'you', 'he', 'she', 'that', 'we', 'they', 'with', 
    'are', 'be', 'at', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'not'
]

    df = pd.read_csv("movies.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
    print(df["score"].unique())

    df = preprocess_dataset(df)

    most_used = write_most_used_words(df)

    df["review_text"] = df["review_text"].apply(lambda x: remove_stopwords(x, most_used))

    ratings = df["score"].unique()
    plot_score_distribution(df)
    write_most_used_words_per_rating(df, ratings)