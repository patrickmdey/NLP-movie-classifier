import math
import pandas as pd
from loguru import logger
from datasets import Dataset
from sklearn.model_selection import train_test_split

def reduce_columns(df, rating_col, review_col, reduce_cols=True):
    """
    Reduces the columns of the dataframe to only the rating and review column.
    """
    logger.info("Reducing columns to only rating and review")
    if reduce_cols:
        df = df[[rating_col, review_col]]
    df = df.rename(columns={rating_col: "rating", review_col: "review"})
    return df

def reduce_ratings(df, col="Rating"):
    df[col] = df[col].apply(lambda x: math.ceil(int(x)/2))
    return df 

def get_bert_dataset(df, text_col, label_col):
    """
        Returns a DatasetDict with the following structure:
        {
            "train": Dataset({
                features: ['text', 'label']
                num_rows: int
            }),
            "test": Dataset({
                features: ['text', 'label']
                num_rows: int
            })
        }
        Args:
            data_path (str): Path to the dataset
    """
    # shuffle rows
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df['label'] = df['label'] - 1
    dataset = Dataset.from_pandas(df)

    # Ensure that the labels are in the correct format (integers)
    dataset = dataset.map(lambda example: {'label': int(example['label'])})

    return dataset.train_test_split(test_size=0.2, shuffle=True)

def get_review_dataset(data_path, separator=";", label_col="rating", text_col="review", num_lines=None):
    
    df = pd.read_csv(data_path, sep=separator, on_bad_lines="skip")
    df = df.sample(num_lines) if num_lines is not None else df
    return get_bert_dataset(df, text_col, label_col)


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
