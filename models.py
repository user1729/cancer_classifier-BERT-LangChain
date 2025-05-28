from transformers import pipeline
import os
import re

class CancerClassifier:
    def __init__(self, model_path: str):
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            return_all_scores=True,
            device=0 if os.environ.get("USE_GPU", "false").lower() == "true" else -1,
        )

    def predict(self, text: str):
        results = self.classifier(text)
        return {
            "predicted_labels": ["Non-Cancer", "Cancer"],
            "confidence_scores": {
                "Non-Cancer": results[0][0]["score"],
                "Cancer": results[0][1]["score"],
            },
        }

class CancerExtractor:
    def __init__(self, model_path ="alvaroalon2/biobert_diseases_ner"):
        self.extractor = pipeline(
            "ner",
            model=model_path,
            aggregation_strategy="simple",
            device=0 if os.environ.get("USE_GPU", "false").lower() == "true" else -1,
        )
        self.cancers = [
            "cancer",
            "astrocytoma",
            "medulloblastoma",
            "meningioma",
            "neoplasm",
            "carcinoma",
            "tumor",
            "melanoma",
            "mesothelioma",
            "leukemia",
            "lymphoma",
            "sarcomas",
        ]

    def predict(self, text: str):
        results = self.extractor(text)
        extractions = self.extract_diseases(results)
        extractions_cleaned = self.clean_diseases(extractions)
        detections = self.detect_cancer(extractions_cleaned)
        return detections

    def extract_diseases(self, entities):
        entities = self.merge_subwords(entities)
        diseases = [
            entity["word"]
            for entity in entities
            if "disease" in entity["entity_group"].lower()
        ]
        return diseases

    def merge_subwords(self, entities):
        merged_entities = []
        current_entity = None
        for entity in entities:
            if current_entity is None:
                current_entity = entity.copy()
            else:
                # Check if this entity is part of the same word as the previous one
                if (
                    entity["start"] == current_entity["end"]
                    and "disease" in entity["entity_group"].lower()
                    and "disease" in current_entity["entity_group"].lower()
                ):
                    # Merge with previous entity
                    current_entity["word"] += entity["word"].replace("##", "")
                    current_entity["end"] = entity["end"]
                    current_entity["score"] = (
                        current_entity["score"] + entity["score"]
                    ) / 2
                else:
                    merged_entities.append(current_entity)
                    current_entity = entity.copy()

        if current_entity is not None:
            merged_entities.append(current_entity)
        return merged_entities

    def clean_diseases(self, text_list):
        text_list = [re.sub(r"[^a-zA-Z]", " ", t) for t in text_list]
        unique_text = set([t.lower() for t in text_list])  # and (t not in stop_words)
        cleaned_text = [
            t for t in unique_text if (3 <= len(t.strip()) <= 50 and ("##" not in t))
        ]
        return cleaned_text

    def detect_cancer(self, text_list):
        detected_cancers = [
            word2.lower()
            for word2 in text_list
            if any(word1.lower() in word2.lower() for word1 in self.cancers)
        ]
        return set(detected_cancers)