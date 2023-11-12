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

    model_type = "bert"
    prompt="I thought this was a 'great' film! It was funny at times and very funny at others, but still got a long way to go! However, I think this is still one of the few films in my collection of films that makes up for the short film running time. I think the film is very well thought out, it\'s very well set up, it\'s very well acted and has very memorable characters. The special effects are definitely top notch - the aliens are"

    if model_type == "roberta":

        checkpoint = "roberta-base"
        device = "cuda"

        logger.info("Loading tokenizer")
        # Load the tokenizer and tokenize the dataset
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        token_id = 1

        logger.info("Loading BERT model")
        model_path = "out/roberta/saved-models/imdb80-roberta"
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5, pad_token_id=token_id).to(device)

        """
        Classify a single prompt
        """
        result = classify(model, prompt, device=device, tokenizer=tokenizer)
        print("Rating:", result)
    else:
        #TODO: device cuda for usage in pampero or cuda
        checkpoint = "distilbert-base-uncased"
        device = "cuda"

        logger.info("Loading tokenizer")
        # Load the tokenizer and tokenize the dataset
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        token_id = tokenizer.pad_token_id

        logger.info("Loading BERT model")
        model_path = "out/bert/saved-models/imdb80-distilbert2"
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5, pad_token_id=token_id).to(device)

        """
        Classify a single prompt
        """
        result = classify(model, prompt, device=device, tokenizer=tokenizer)
        print("Rating:", result)