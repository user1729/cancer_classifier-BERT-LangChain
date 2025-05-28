import re
import nltk
import pandas as pd
from data import CancerDataset
from transformers import pipeline


class CancerExtractionPipeline:
    """End-to-end cancer classification pipeline"""

    def __init__(
        self,
        ner_model: str = "alvaroalon2/biobert_diseases_ner",
        dataset_path: str = "dataset",
    ):
        self.ner_model = ner_model
        self.dataset_path = dataset_path
        self.dataset = CancerDataset(self.dataset_path)
        preprocessed_data = self.dataset.prepare_datasets()
        self.target_data = preprocessed_data["target_dataset"]
        self.cancers = self.dataset.cancers
        self.ner_pipeline = pipeline(
            "ner",
            model=self.ner_model,  # Alternative model
            aggregation_strategy="simple",
        )

    def get_extraction_pipeline(self):
        return self.ner_pipeline

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

    def extract_diseases(self, text):
        entities = self.ner_pipeline(text)
        entities = self.merge_subwords(entities)
        # print(entities)
        # Filter for disease entities and get their text
        diseases = [
            entity["word"]
            for entity in entities
            if "disease" in entity["entity_group"].lower()
        ]
        return diseases

    # Download stopwords once
    nltk.download("stopwords")

    def clean_diseases(self, text_list):
        # stop_words = set(stopwords.words('english'))
        text_list = [re.sub(r"[^a-zA-Z]", " ", t) for t in text_list]
        # text_list.extend([word.strip() for word in text_list.split(',') if word.strip()])
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
