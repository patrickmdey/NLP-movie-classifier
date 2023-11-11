from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import *
from loguru import logger


def classify(model, prompt, device, tokenizer):
    """
    Classifies a prompt using the given model and tokenizer.
    Args:
        -  model: The model to use for classification
        -  prompt: The prompt to classify
        -  device: The device to use for classification
        -  tokenizer: The tokenizer to use for classification
    Returns:
        -  The predicted rating
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt",truncation=True).to(device)
    model.eval()
    outputs = model(input_ids).logits

    print(model(input_ids).logits)

    return outputs.argmax(-1).item() + 1 # Add 1 because the ratings are 1-5 and the model outputs 0-4


if __name__ == "__main__":

    #TODO: device cuda for usage in pampero or cuda
    checkpoint = "distilbert-base-uncased"
    device = "cuda"

    logger.info("Loading tokenizer")
    # Load the tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    logger.info("Loading BERT model")
    model_path = "out/bert/saved-models/imdb80-distilbert"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5, pad_token_id=tokenizer.pad_token_id).to(device)

    """
    Classify an entire dataset
    """
    df = pd.read_csv("out/generator/joined_movies_preprocessed/generated_reviews_classified.csv", sep=";")
    logger.info("Classifying prompt")
    df["predicted"] = df["review"].apply(lambda x: classify(model, x, device=device, tokenizer=tokenizer))
    df = df[['predicted', 'rating', 'review']]
    df.to_csv("out/generator/joined_movies_preprocessed/generated_reviews_classified.csv", sep=";", index=False)

    """
    Classify a single prompt
    """
    # prompt="I thought this was a 'great' film! It was funny at times and very funny at others, but still got a long way to go! However, I think this is still one of the few films in my collection of films that makes up for the short film running time. I think the film is very well thought out, it\'s very well set up, it\'s very well acted and has very memorable characters. The special effects are definitely top notch - the aliens are"
    # result = classify(model, prompt, device=device, tokenizer=tokenizer)
    # print("Rating:", result)
