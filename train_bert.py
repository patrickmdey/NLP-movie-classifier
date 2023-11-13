import torch
from sklearn.discriminant_analysis import softmax as sk_softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from utils import *
from datetime import datetime
from loguru import logger
import time
import os


class MyCallback(TrainerCallback):
    def __init__(self, tokenizer, device):
        self.epochs = 0
        self.device = device
        self.tokenizer = tokenizer
        """
        reviews are the 20% of the dataset that was not used for training
        """
        df = pd.read_csv(
            "datasets/imdb_movies_preprocessed_test.csv", sep=";", encoding="utf-8")
        df.rename(columns={"review": "text", "rating": "label"}, inplace=True)
        self.reviews = df.to_dict("records")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("Finished epoch {}".format(self.epochs))
        self.epochs += 1
        # Access current model
        current_model = kwargs["model"]

        if not os.path.exists("out/bert/progress"):
            os.makedirs("out/bert/progress")
            with open("out/bert/progress/classify.csv", "w") as f:
                f.write("epoch;text;label;prediction\n")

        with open("out/bert/progress/classify.csv", "a") as f:
            for review in self.reviews:
                try:
                    input_ids = self.tokenizer.encode(
                        review["text"], return_tensors="pt", truncation=True, max_length=512).to(self.device)
                    current_model.eval()
                    outputs = current_model(input_ids).logits

                    logger.info(f"Logits: {outputs}")

                    cpu_outputs = outputs.cpu()

                    softmax_probs = sk_softmax(cpu_outputs.detach().numpy())

                    logger.info(f"Softmax: {softmax_probs}")

                    f.write(
                        f"{self.epochs};{review['text']};{review['label']};{softmax_probs.argmax(-1).item() + 1}\n")
                except Exception as e:
                    logger.error("Error classifying review, skipping...")

def tokenize_dataset(tokenizer, dataset):
    def tokenize_function(batch):
        tokenized = tokenizer(batch["text"], padding=True, truncation=True)
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], "label": batch["label"]}

    # Tokenize the dataset
    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    test_dataset = dataset["test"].map(tokenize_function, batched=True)

    return {"train": train_dataset, "test": test_dataset}


def main():
    start_time = time.time()
    checkpoint = "distilbert-base-uncased"
    dataset_path = "datasets/imdb_movies_preprocessed_train.csv"
    device = "cuda"

    logger.info("Loading dataset")
    dataset = get_review_dataset(dataset_path, separator=";",
                                 text_col="review", label_col="rating")

    # Load the tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # sets the pad token to be ignored when computing the metrics for the training
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenizes the dataset returning a dictionary of two datasets: train and test
    # Each dataset has the following features: text, label, input_ids and attention_mask
    logger.info("Tokeinizing dataset")
    tokenized_datasets = tokenize_dataset(tokenizer, dataset)
    logger.info("Finished tokenizing dataset")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load the model
    logger.info("START | Loading bert model")
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=5,
        pad_token_id=tokenizer.pad_token_id).to(device)
    logger.info("END | Loading bert model")

    # Define the training arguments
    logger.info("Setting training arguments")
    training_args = TrainingArguments(
        evaluation_strategy="epoch",  # eval en validation set
        num_train_epochs=5,
        log_level="error",
        output_dir="out/bert/imdb80-checkpoints/" + \
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        save_strategy="epoch",
        fp16=True,  # Only works with cuda
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        optim="adamw_torch",
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        weight_decay=0.1,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        disable_tqdm=True,

        overwrite_output_dir=True,
        # forma de regularizacion (restringe el tamaño de updates de SGD)
        # warmup evita divergencia de loss en primeros steps (10%)
        warmup_ratio=0.1,
        do_eval=True,  # eval en validation set
    )

    callback = MyCallback(tokenizer=tokenizer, device=device)

    logger.info("Setting trainer")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        # TODO: check i think its not using it
        eval_dataset=tokenized_datasets["test"],
        callbacks=[callback]
    )

    logger.info("Start Training")
    train_output = trainer.train()
    logger.info("End Training")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    trainer.save_model("out/bert/saved-models/imdb80-distilbert")

    end_time = time.time()

    # Calcula el tiempo transcurrido en segundos
    elapsed_time = end_time - start_time
    logger.info(elapsed_time)


if __name__ == "__main__":
    main()
