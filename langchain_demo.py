from langchain.chains import SequentialChain, TransformChain
from typing import Dict, Any, List, Union
from transformers import AutoTokenizer
import logging

from models import CancerClassifier, CancerExtractor
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProcessingError(Exception):
    """Custom exception for model processing errors"""
    pass

# Initialize your models with error handling
try:
    classification_pipeline = CancerClassifier("models/fine_tuned")
    extraction_pipeline = CancerExtractor()

except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

def batch_classification_transform(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Process batch of texts through classification model"""
    try:
        texts = inputs["input_texts"]
        if isinstance(texts, str):
            texts = [texts]  # Convert single text to batch of one
            
        results = []
        for text in texts:
            try:
                result = classification_pipeline.predict(text)
                results.append(str(result))
            except Exception as e:
                logger.warning(f"Classification failed for text: {text[:50]}... Error: {str(e)}")
                results.append({"error": str(e)})
                
        return {"classification_results": results}
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        raise ModelProcessingError(f"Classification processing error: {str(e)}")

def batch_extraction_transform(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Process batch of classification results through extraction model"""
    try:
        texts = inputs["input_texts"]
        if isinstance(texts, str):
            texts = [texts]  # Convert single text to batch of one
            
        extraction_results = []
        for text in texts:
            try:
                result = extraction_pipeline.predict(text)
                extraction_results.append(str(result))
            except Exception as e:
                logger.warning(f"Extraction failed for input: {str(text)[:50]}... Error: {str(e)}")
                extraction_results.append({"error": str(e)})
                
        return {"extraction_results": extraction_results}
    except Exception as e:
        logger.error(f"Batch extraction failed: {str(e)}")
        raise ModelProcessingError(f"Extraction processing error: {str(e)}")

# Create the processing chains
classification_chain = TransformChain(
    input_variables=["input_texts"],
    output_variables=["classification_results"],
    transform=batch_classification_transform
)

extraction_chain = TransformChain(
    input_variables=["input_texts"],
    output_variables=["extraction_results"],
    transform=batch_extraction_transform
)

# Create the sequential chain
processing_chain = SequentialChain(
    chains=[classification_chain, extraction_chain],
    input_variables=["input_texts"],
    output_variables=["classification_results", "extraction_results"],
    verbose=True
)

def process_texts(texts: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Process one or multiple texts through the pipeline
    Args:
        texts: Single text string or list of texts
    Returns:
        Dictionary with classification and extraction results
    """
    try:
        if isinstance(texts, str):
            texts = [texts]
            
        if not isinstance(texts, list):
            raise ValueError("Input must be string or list of strings")
            
        return processing_chain({"input_texts": texts})
    except Exception as e:
        logger.error(f"Processing pipeline failed: {str(e)}")
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Single text processing
    single_result = process_texts("cancer is a life-threatening disease")
    print("Single result:", single_result)
    
    # Batch processing
    multiple_texts = [
        "This study investigates novel biomarkers for early detection of lung cancer in non-smokers. Patients with breast cancer and melanoma showed improved outcomes.",
        "Breast cancer is a disease where cells in the breast tissue grow out of control, forming tumors. It's the most common cancer in women, and while less common in men, it can occur in both sexes.",
        "Eye twitching, also known as a myokymia or eyelid twitch, is a common and usually harmless condition where the eyelid muscle spasms involuntarily. It's often triggered by lifestyle factors like stress, fatigue, or excessive caffeine.",
    ]
    batch_result = process_texts(multiple_texts)
    print("Batch results:")
    for i, (class_res, extract_res) in enumerate(zip(
        batch_result["classification_results"],
        batch_result["extraction_results"]
    )):
        print(f"\nText {i+1}: " + multiple_texts[i])
        print("Classification:", class_res)
        print("Extraction:", extract_res)
        print("\n")

