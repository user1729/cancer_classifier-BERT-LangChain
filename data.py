import os
import re
import string
import chardet
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from bs4 import BeautifulSoup, Comment
from sklearn.model_selection import train_test_split
from datasets import Dataset, concatenate_datasets, load_dataset


@dataclass
class CancerDataset:
    """
    A class to handle cancer/non-cancer classification datasets
    with efficient data loading and preprocessing.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path  #'dataset'
        self.label_map = {0: "Non-Cancer", 1: "Cancer"}
        # Medical-specific regex patterns
        self.patterns = {
            "copyright": re.compile(r"Copyright ©.*?\.", re.IGNORECASE),
            "url": re.compile(r"https?://\S+|www\.\S+"),
            "email": re.compile(r"\b\S+@\S+\.\S+\b"),
            "references": re.compile(r"references\s*:\s*\S+", re.IGNORECASE),
            "citations": re.compile(r"\\cite\{.*?\}", re.IGNORECASE),
            "pmid": re.compile(r"\(PMID: \d+\)", re.IGNORECASE),
            "doi": re.compile(r"^DOI:.*\.\s*$", re.IGNORECASE),
            "empty_line": re.compile(r"^\s*\n", re.MULTILINE),
        }

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

    def detect_encoding(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            rawdata = f.read(10000)  # Sample first 10KB
            return chardet.detect(rawdata)["encoding"]

    ## Load Data & Handle Missing Abstracts
    def parse_file(self, file_path: str, encoding: str) -> Dict:
        # Extract ID, Title, Abstract (skip if missing)
        try:
            with open(
                file_path, "r", encoding=encoding
            ) as f:  # if needed errors='ignore'
                content = f.read()
            pmid = re.search(r"<ID:(\d+)>", content).group(1)
            title = (
                re.search(r"Title: (.+?)\nAbstract:", content, re.DOTALL)
                .group(1)
                .strip()
            )
            abstract = re.search(r"Abstract: (.+)", content, re.DOTALL).group(1).strip()
            return {"pmid": pmid, "title": title, "abstract": abstract}
        except Exception as e:
            ## Skip corrupted/missing abstracts, print exceptions
            # print(f"Error in {file_path}: {str(e)}")
            return None

    def load_target_dataset(self) -> pd.DataFrame:
        data = []
        for label, item in self.label_map.items():
            dir_path = f"{self.dataset_path}/{item}"
            for file in os.listdir(dir_path):
                encoding = self.detect_encoding(f"{dir_path}/{file}")
                parsed = self.parse_file(f"{dir_path}/{file}", encoding=encoding)
                if parsed:
                    parsed["label"] = label
                    data.append(parsed)
        df = pd.DataFrame(data)
        # Clean URLs, and different patterns
        df["abstract"] = df["abstract"].apply(self.remove_patterns)
        # Clean html formatting
        df["abstract"] = df["abstract"].apply(self.html_to_text)
        # Filter rows where abstract length >= 200
        df = df[df["abstract"].str.len() >= 200].reset_index(drop=True)
        # handle missing abstract
        df.dropna(subset=["abstract"], inplace=True)
        print(f"Loaded {len(df)}/1000 abstracts. Dropped {1000 - len(df)}.")
        return df

    def process_ncbi_disease(self, example: str) -> Dict:
        text = " ".join(example["tokens"])
        is_cancer = 0
        if any(tag in [1, 2] for tag in example["ner_tags"]):
            if any(keyword.lower() in text.lower() for keyword in self.cancers):
                is_cancer = 1
        return {"abstract": text, "label": int(is_cancer)}

    def process_bionlp(self, example: str) -> Dict:
        is_cancer = (
            any(
                annotation["type"] == "Cancer"
                for annotation in example["text_bound_annotations"]
            )
            if example["text_bound_annotations"]
            else False
        )

        return {"abstract": example["text"], "label": int(is_cancer)}

    def process_bc5cdr(self, example: str) -> Dict:
        is_cancer = 0
        cancer_entities = []
        passage = [p for p in example["passages"] if p["type"] == "abstract"][0]
        for entity in passage["entities"]:
            if entity["type"] == "Disease":
                text = " ".join(entity["text"]).lower()
                if any(keyword.lower() in text for keyword in self.cancers):
                    is_cancer = 1
        return {
            "abstract": passage["text"],  # Combine title + abstract
            "label": int(is_cancer),
        }

    def get_label_mapping(self) -> Dict:
        """Get label to text mapping"""
        return self.label_map

    def prepare_datasets(self):
        """Prepare all datasets with preprocessing"""
        target_dataset_df = self.load_target_dataset()
        ncbi = load_dataset("ncbi_disease", trust_remote_code=True).map(
            self.process_ncbi_disease, remove_columns=["id", "tokens", "ner_tags"]
        )
        bionlp = load_dataset("bigbio/bionlp_st_2013_cg", trust_remote_code=True).map(
            self.process_bionlp,
            remove_columns=[
                "id",
                "document_id",
                "text_bound_annotations",
                "events",
                "relations",
                "equivalences",
                "attributes",
                "normalizations",
            ],
        )
        bc5cdr = load_dataset("bigbio/bc5cdr", trust_remote_code=True).map(
            self.process_bc5cdr, remove_columns=["passages"]
        )

        train_df, test_df = train_test_split(
            target_dataset_df[["abstract", "label"]], test_size=0.2, random_state=10
        )
        target_train_dataset = Dataset.from_pandas(train_df)
        target_test_dataset = Dataset.from_pandas(test_df)

        # Combine all datasets
        combined_dataset = concatenate_datasets(
            [target_train_dataset, ncbi["train"], bionlp["train"], bc5cdr["train"]]
        ).shuffle(seed=10)

        return {
            "test": target_test_dataset,
            "train": combined_dataset,
            "target_dataset": target_dataset_df,
        }

    def html_to_text(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        # Remove navigation/header/footer sections (optional — heuristic based)
        for tag in soup.find_all(["nav", "header", "footer", "aside"]):
            tag.decompose()
        # Remove <a> tags (strip links)
        for a_tag in soup.find_all("a"):
            a_tag.unwrap()
        # Get text content
        pure_text = soup.get_text(separator=" ", strip=True)
        # Collapse multiple spaces
        pure_text = " ".join(pure_text.split())
        return pure_text

    def clean_dataframe(
        self, df: pd.DataFrame, text_column: str = "abstract"
    ) -> pd.DataFrame:
        """Clean a pandas DataFrame column"""
        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)
        return df

    def remove_patterns(self, text: str) -> str:
        """Remove medical-specific patterns"""
        for pattern in self.patterns.values():
            text = pattern.sub("", text)
        return text
