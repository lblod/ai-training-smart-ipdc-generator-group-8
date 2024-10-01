import csv
import logging
import time
from pathlib import Path
import mlflow
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import pandas as pd


DATAFILE = "producten_en_diensten_2024-09-13_21-47-37.csv"
HUGGINGFACE_MODEL = "papluca/xlm-roberta-base-language-detection"
ML_FLOW_URI = "http://localhost:5000"


def load_data():
    count = 0
    buffer = []
    with open(Path.home() / "producten_en_diensten_2024-09-13_21-47-37.csv" , 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) == 47:
                buffer.append({
                    'thema': row[5],
                    'tpe': row[7],
                    'beschrijving': row[1]
                })
    return buffer[1:]


def preprocess_function(example):
    text = example['beschrijving']
    all_labels = example['thema'].split(', ')
    labels = [0. for i in range(len(classes))]
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.

    example = tokenizer(text, truncation=True)
    example['labels'] = labels
    return example


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def compute_metrics(eval_pred):
   predictions, labels = eval_pred
   predictions = sigmoid(predictions)
   predictions = (predictions > 0.5).astype(int).reshape(-1)
   return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


if __name__ == '__main__':
    try:
        start_time = time.time()

        # mlflow.set_tracking_uri(ML_FLOW_URI)
        # mlflow.set_experiment('thema-ipdc-model')

        buffer = load_data()
        df = pd.DataFrame(buffer)

        dataset = Dataset.from_pandas(df[df.thema != '']).train_test_split(test_size=0.15)

        classes = df.thema.str.get_dummies(sep=', ').columns
        class2id = {class_: id for id, class_ in enumerate(classes)}
        id2class = {id: class_ for class_, id in class2id.items()}

        tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
        tokenized_dataset = dataset.map(preprocess_function)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        model = AutoModelForSequenceClassification.from_pretrained(
            HUGGINGFACE_MODEL,
            num_labels=len(classes),
            id2label=id2class,
            label2id=class2id,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir="thema_ipdc_model",
            learning_rate=2e-5,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            num_train_epochs=1,
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        #
        duration = time.time() - start_time  # Calculate the duration
        print(f"Done. Model training ran for  {duration:.2f} seconds.")

    except Exception as ex:
        logging.error(ex)
