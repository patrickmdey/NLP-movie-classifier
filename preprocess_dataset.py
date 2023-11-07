import json
import pandas as pd
import re
from loguru import logger
from tqdm import tqdm
from stop_words import get_stop_words
from utils import reduce_columns, reduce_ratings

def get_symbols(df, out_name="symbols"):
    logger.info("Getting all symbols from reviews")
    symbols = []
    for rev in df["review"]:
        symbols += re.findall(r"[^a-zA-Z0-9,.!?.\"\']+", rev)
    symbols = list(set(symbols))
    pd.DataFrame(symbols, columns=["symbols"]).to_csv(f"datasets/{out_name}.csv", index=False)

def _remove_symbols_from_row(row):
    permited_symbols = ["!", "?", ".", ","]
    row = re.sub(r"<[^>]*>", "", row) # Remove html tags
    row = re.sub(r";", ",", row) # Change ; to ,
    row = re.sub(r"[^a-zA-Z0-9\.,?!'\(\) ]+", "", row) # Remove all symbols except . , ? ! ' ( )
    # row = re.sub(r"\.\s+{1,}", ".", row) # TODO: CHECK
    row =  re.sub(r"\.\s+", ".", row)    
    # row = re.sub(r". ", "", row)
    for symbol in permited_symbols:
        row = re.sub(rf"[{symbol}]+", symbol, row)
    # row = re.sub(r"[.]+", ".", row) # Trim multiple points to one
    
    return row

def remove_symbols_from_reviews(df):
    """
    Remove all symbols from the reviews except points, commas, question marks and exclamation marks.
    Trims multiple spaces to one.
    Trims multiple symbols to one.
    """
    logger.info("Removing and trimming symbols from reviews")
    get_symbols(df)
    df["review"] = df["review"].apply(lambda x: _remove_symbols_from_row(x))
    return df

def remove_unwanted(df, rating_col, review_col):
    logger.info(f"Removing rows with Null in {rating_col}")
    df = df[df[rating_col] != "Null"]
    df = df[~df[review_col].str.contains('\*')]
    return df
    
def balance_reviews(df, max_amount=None):
    """
    Balances the reviews by taking the amount of the lowest rating and 
    taking the same amount of reviews from each rating.
    """
    min_amount = int(df['rating'].value_counts().min())
    if max_amount:
        min_amount = min(min_amount, max_amount)
    logger.info(f"Balancing reviews to {min_amount} per rating")
    df["rating"] = df.apply(lambda x: int(x["rating"]), axis=1)
    ratings = df["rating"].unique()
    new_df = pd.DataFrame(columns=['rating', 'review'])
    for rating in ratings:
        rating_samples = df[df['rating'] == rating].sample(min_amount)
        new_df = pd.concat([new_df, rating_samples], ignore_index=True)
    
    return new_df

def _remove_words(review, stop_words, pbar, lower_case=False):
    pbar.update(1)
    # If lower_case is True, convert the review to lowercase
    if lower_case:
        review = review.lower()

    # Split the review into words
    words = review.split()

    # Filter out words that are in the stop_words list
    cleaned_words = [word for word in words if word.lower() not in stop_words]

    # Join the cleaned words back into a string
    return ' '.join(cleaned_words)

def remove_stop_words(df, stop_words=None, lower_case=False):
    if not stop_words:
        logger.info("Using default stop words")
        # stop_words = list(pd.read_csv("datasets/stopwords.csv")["words"])
        stop_words = get_stop_words('en')
    logger.info("Removing stop words")

    pbar = tqdm(total=len(df))
    df["review"] = df["review"].apply(lambda x: _remove_words(x, stop_words, pbar, lower_case=lower_case))
    return df
    

if __name__ == "__main__":
    """
    This script preprocesses the dataset according to the config.json file.
    Example:
    {
        "dataset": "stanford.csv",
        "balance": true,
        "clean": true,
        "csv_sep": ",",
        "rating_col": "Rating",
        "review_col": "Review",
        "reduce_rating": true
    }
    """
    config = json.load(open("config_preprocess.json"))
    dataset_path = config["dataset"]
    reduce_cols = config["reduce_cols"] if "reduce_cols" in config else False
    balance = config["balance"] if "balance" in config else False
    clean = config["clean"] if "clean" in config else False
    lower_case = config["lower_case"] if "lower_case" in config else False
    csv_sep = config["csv_sep"] if "csv_sep" in config else ";"
    rating_col = config["rating_col"] if "rating_col" in config else "rating"
    review_col = config["review_col"] if "review_col" in config else "review"
    reduce_rating = config["reduce_rating"] if "reduce_rating" in config else False

    logger.info(f"Preprocessing dataset {dataset_path}")
    df = pd.read_csv(dataset_path, sep=csv_sep, on_bad_lines='skip')
    df = df.dropna()
    df = remove_unwanted(df, rating_col, review_col)
    # TODO: remove reviews with asterisks and remove emojis.
    # TODO: prueba solo con \u
    # TODO: remove http links
    df = reduce_columns(df, rating_col, review_col, reduce_cols)

    def _trim_spaces(text):
        words = text.split()
        return " ".join(words)

    df["review"] = df["review"].apply(lambda x: _trim_spaces(x)) # Remove trailing spaces
    df = remove_symbols_from_reviews(df)
    
    if balance:
        try:
            df = balance_reviews(df)
        except:
            logger.warning("Could not balance dataset because the ratings are not integers")
    if clean:
        df = remove_stop_words(df, lower_case=lower_case)
    
    if reduce_rating:
        try:
            df = reduce_ratings(df, col="rating")
        except Exception as e:
            logger.error(e)
            logger.warning("Could not reduce ratings because the ratings are not integers")
    
    new_path = dataset_path.split(".")[0] + "_preprocessed.csv"
    df.to_csv(new_path, index=False, sep=csv_sep)



