import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from utils import *
from datetime import datetime
from loguru import logger
import time
import os

#TODO: check the reviews fot evaluation
class MyCallback(TrainerCallback):
    def __init__(self, tokenizer, device):
        self.epochs = 0
        self.device = device
        self.tokenizer = tokenizer
        self.reviews = [{
            "text": "after all the hype over this movie, i was expecting something great. a disappointment!!!  i found myself fast-forwarding through parts because it was soooo boring.  the story was mediocre, and the acting wasn'tbetter.  really, after the sex scene, i wasready to be done with this movie. it came across as violent, and that left me queasy (and fyi,itbeen a man and woman in the exact same scene, ithave bothered meas much).save your money...not worth it.",
            "label": 1
        }, {
            "text": "at first i was enjoying the movie. but after a while,things weren't making sense and i thought thatwere going to be explained in the end, but it didn't happen. a creepy guy drops off a box and tells cameron diaz's character thatshe presses the button shenotget $1,000,000 and someonedoesn't knowdie (andshouldn't tell anyone excepthusband) but he omitted to tellthat by pressing the buttoncanenjoy the moneyshelive withson being deaf and blindher husbandto killin order forson to be normal.it wasn'tabout having the power to make a random person die and getting the money,life wasstake. consequently, the creepy guyprogram the box for the next victim and tell thepart to the womanmost likelya husband andpeopledie. to be honest the concept of the story for a horror movie is good, but ithink it livedto the vision of the directorwhoever camewith the story and forreason due tosort of pressuresomething, the director decided to go with it...i'mspeculating.notthe actorsterrible.",
            "label": 2
        },  {
            "text": "the movie itself is brilliant, original, funny, perfectly cast and ontop-10 list. however,you have the originalwaste your $$ on the blue-ray. on the blue-ray special features is a discussionbudget compromises ritchie and the photographerforced to makeshooting on 16mm. on the originalit blends with the movie and works. on blue-ray its distracting and jarring. on balance and inopinion, the blue-ray is inferior to the original.",
            "label": 3
        }, {
            "text": "the first half is fresh and fun, full of unexpected characterizations thatnever be caught dead (like a chicken) in a disney film. the second half devolves into numerous disney-esque plot contrivances, none of which igo into becausemight be spoilers to some. i wish the storyhave finished with the same energy with which it began instead of falling flat. i loved everything else this film. a solid four stars that could have (and should have) been five.",
            "label": 4
        }, {
            "text": "great older movie - clean entertainment for the family, the baby boomers remember movies made to be watched, and watched over.  truly enjoyed seeing it again.",
            "label": 5
        }]

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.epochs += 1
        # Access current model
        current_model = kwargs["model"]

        if not os.path.exists("out/bert-epoch-progress"):
            os.makedirs("out/bert-epoch-progress")
            with open("out/bert-epoch-progress/classify.csv", "w") as f:
                f.write("epoch;text;label;prediction\n")

        with open("out/bert-epoch-progress/classify.csv", "a") as f:
            for review in self.reviews:
                input_ids = self.tokenizer.encode(
                    review["text"], return_tensors="pt").to(self.device)
                current_model.eval()
                outputs = current_model(input_ids).logits
                logger.info("Review: {}, Label: {}, Prediction: {}".format(
                    review["text"], review["label"], outputs.argmax(-1).item() + 1))
                f.write(
                    f"{self.epochs};{review['text']};{review['label']};{outputs.argmax(-1).item() + 1}\n")


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
    device = "cpu"

    logger.info("Loading dataset")
    dataset = get_review_dataset(dataset_path, separator=";",
                                 text_col="review", label_col="rating")

    # Load the tokenizer and tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # sets the pad token to be ignored when computing the metrics for the training

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
        output_dir="out/bert/standford-checkpoints/" + \
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        save_strategy="epoch",
        # fp16=True, #Only works with cuda
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
        eval_dataset=tokenized_datasets["test"], #TODO: check i think its not using it
        callbacks=[callback]
    )

    logger.info("Start Training")
    train_output = trainer.train()
    logger.info("End Training")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    trainer.save_model("out/bert/saved-models/standford-distilbert")

    end_time = time.time()

    # Calcula el tiempo transcurrido en segundos
    elapsed_time = end_time - start_time
    logger.info(elapsed_time)

if __name__ == "__main__":
    main()
