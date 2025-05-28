import torch
import json
import evaluate
import numpy as np
from data import CancerDataset
from sklearn.metrics import confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)


class CancerClassifierPipeline:
    """End-to-end cancer classification pipeline"""

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        dataset_path: str = "dataset",
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        self.dataset_path = dataset_path
        self.dataset = CancerDataset(self.dataset_path)
        preprocessed_data = self.dataset.prepare_datasets()
        self.train_data = preprocessed_data["train"]
        self.test_data = preprocessed_data["test"]

    def tokenize(self, examples):
        return self.tokenizer(
            examples["abstract"], padding="max_length", truncation=True, max_length=256
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = evaluate.load("accuracy").compute(
            predictions=predictions, references=labels
        )["accuracy"]
        f1 = evaluate.load("f1").compute(
            predictions=predictions, references=labels, average="weighted"
        )["f1"]

        return {"accuracy": accuracy, "f1": f1}

    def train(self):
        """Fine-tune the model"""
        train_data_tokenized = self.train_data.map(self.tokenize, batched=True)
        split_dataset = train_data_tokenized.train_test_split(test_size=0.2)
        train_data_split = split_dataset["train"]
        val_data_split = split_dataset["test"]

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, problem_type="single_label_classification"
        )

        training_args = TrainingArguments(
            output_dir="cancer_classifier",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            # fp16=True,  # Enable mixed precision
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data_split,
            eval_dataset=val_data_split,
            compute_metrics=self.compute_metrics,
        )

        print("-" * 50)
        print("Training...")

        self.trainer.train()
        self.trainer.save_model("models/fine_tuned")

        print("...Finished training")
        print("-" * 50)

    def evaluate_model(self, model_path: str = None):
        """Evaluate model performance"""
        if model_path is None:
            model = self.trainer.model
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

        tokenized_test = self.test_data.map(self.tokenize, batched=True)

        trainer = Trainer(model=model)
        predictions = trainer.predict(tokenized_test)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        # Generate metrics
        cm = confusion_matrix(labels, preds)
        report = classification_report(
            labels, preds, target_names=["Non-Cancer", "Cancer"], output_dict=True
        )

        # Format confusion matrix for display
        formatted_cm = {
            "Actual Cancer": {
                "Predicted Cancer": int(cm[1][1]),
                "Predicted Non-Cancer": int(cm[1][0]),
            },
            "Actual Non-Cancer": {
                "Predicted Cancer": int(cm[0][1]),
                "Predicted Non-Cancer": int(cm[0][0]),
            },
        }

        return {
            "accuracy": report["accuracy"],
            "f1_score": report["weighted avg"]["f1-score"],
            "confusion_matrix": formatted_cm,
            "classification_report": report,
        }

    def predict(self, text: str, model_path: str = None):
        """Make prediction on single abstract"""
        if model_path is None:
            model_path = "models/fine_tuned"

        classifier = pipeline(
            "text-classification",
            tokenizer=self.tokenizer,
            model=model_path,
            top_k=None #return_all_scores=True,
        )

        results = classifier(text)
        confidence_scores = {
            "Cancer": results[0][1]["score"],
            "Non-Cancer": results[0][0]["score"],
        }

        return {
            "predicted_labels": ["Cancer", "Non-Cancer"],
            "confidence_scores": confidence_scores,
        }
