from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl, DataCollatorWithPadding
from datetime import datetime
from loguru import logger
import time
import os
import torch
import pandas as pd
from utils import get_review_dataset


class MyCallback(TrainerCallback):
    def __init__(self, tokenizer, device, reviews):
        self.epochs = 0
        self.device = device
        self.tokenizer = tokenizer
        self.reviews = reviews

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.epochs += 1
        current_model = kwargs["model"]

        if not os.path.exists("out/t5-epoch-progress"):
            os.makedirs("out/t5-epoch-progress")
            with open("out/t5-epoch-progress/classify.csv", "w") as f:
                f.write("epoch;text;label;prediction\n")

        with open("out/t5-epoch-progress/classify.csv", "a") as f:
            for review in self.reviews:
                input_ids = self.tokenizer(
                    review["text"], return_tensors="pt").input_ids.to(self.device)
                current_model.eval()
                outputs = current_model.generate(input_ids)
                prediction = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                logger.info("Review: {}, Label: {}, Prediction: {}".format(
                    review["text"], review["label"], prediction))
                f.write(
                    f"{self.epochs};{review['text']};{review['label']};{prediction}\n")

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
    checkpoint = "t5-base"
    dataset_path = "datasets/imdb_movies_preprocessed_train.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading dataset")
    # Assuming `get_review_dataset` is a function that loads your dataset
    dataset = get_review_dataset(dataset_path, separator=";",
                                 text_col="review", label_col="rating")

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    # Tokenizes the dataset
    logger.info("Tokenizing dataset")

    tokenized_datasets = tokenize_dataset(tokenizer, dataset)

    # tokenized_datasets = tokenizer(
    #     dataset["train"]["text"], padding=True, truncation=True, return_tensors="pt")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load the T5 model
    logger.info("START | Loading T5 model")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
    logger.info("END | Loading T5 model")

    # Define the training arguments
    logger.info("Setting training arguments")
    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        num_train_epochs=5,
        log_level="error",
        output_dir="out/t5/epoch-checkpoints/" +
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        save_strategy="epoch",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        learning_rate=2e-5,
        weight_decay=0.1,
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        disable_tqdm=True,
        overwrite_output_dir=True,
        warmup_ratio=0.1,
        do_eval=True,
    )

    df = pd.read_csv("datasets/imdb_movies_preprocessed_test.csv", sep=";", encoding="utf-8")
    df.rename(columns={"review": "text", "rating": "label"}, inplace=True)
    reviews = df.to_dict("records")

    callback = MyCallback(tokenizer=tokenizer, device=device, reviews=reviews)

    logger.info("Setting trainer")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets,
        callbacks=[callback]
    )

    logger.info("Start Training")
    train_output = trainer.train()
    logger.info("End Training")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    trainer.save_model("out/t5/saved-models/t5-sequence-classification")

    end_time = time.time()

    elapsed_time = end_time - start_time
    logger.info(elapsed_time)


if __name__ == "__main__":
    main()
