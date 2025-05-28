import json
import argparse
from data import CancerDataset
from train import CancerClassifierPipeline
from extraction import CancerExtractionPipeline


def extract_cancer_from_abstract(extractionpipeline):
    df_copy = extractionpipeline.target_data.copy()
    df_copy["extractions"] = df_copy["abstract"].apply(
        extractionpipeline.extract_diseases
    )
    df_copy["extractions_cleaned"] = df_copy["extractions"].apply(
        extractionpipeline.clean_diseases
    )
    df_copy["detections"] = df_copy["extractions_cleaned"].apply(
        extractionpipeline.detect_cancer
    )
    count = 0
    # Print only abstracts with found diseases
    for idx, row in df_copy[df_copy["detections"].str.len() > 0].iterrows():
        print(f"PMID: {row['pmid']}")
        print(f"Cancers found: {', '.join(row['detections'])}")
        print(f"Diseases found: {', '.join(row['extractions_cleaned'])}")
        print("Abstract excerpt:", row["abstract"][:200] + "...")
        print("-" * 80)
        count = count + 1
    print("Number of successful extractions:", count)
    return df_copy


def classifer_train_and_evaluate(classifierpipeline):
    """Demo script showing before/after fine-tuning comparison"""

    print("\n--- Evaluating Baseline Model ---")
    baseline_results = classifierpipeline.evaluate_model(classifierpipeline.model_name)
    print(baseline_results)
    print(f"Accuracy: {baseline_results['accuracy']:.2f}")
    print(f"F1-score: {baseline_results['f1_score']:.2f}")
    print("Confusion Matrix:\n")
    confusion_matrix = baseline_results["confusion_matrix"]
    print(f"{'':<20}{'Predicted Cancer':<20}{'Predicted Non-Cancer':<20}")
    for actual_label, predictions in confusion_matrix.items():
        row = f"{actual_label:<20}"
        row += f"{predictions['Predicted Cancer']:<20}"
        row += f"{predictions['Predicted Non-Cancer']:<20}"
        print(row)
    # print(json.dumps(baseline_results['confusion_matrix'], indent=2))

    print("\n--- Fine-tuning Model ---")
    classifierpipeline.train()

    print("\n--- Evaluating Fine-Tuned Model ---")
    fine_tuned_results = classifierpipeline.evaluate_model()
    print(f"Accuracy: {fine_tuned_results['accuracy']:.2f}")
    print(f"F1-score: {fine_tuned_results['f1_score']:.2f}")
    print("Confusion Matrix:\n")
    confusion_matrix = fine_tuned_results["confusion_matrix"]
    print(f"{'':<20}{'Predicted Cancer':<20}{'Predicted Non-Cancer':<20}")
    for actual_label, predictions in confusion_matrix.items():
        row = f"{actual_label:<20}"
        row += f"{predictions['Predicted Cancer']:<20}"
        row += f"{predictions['Predicted Non-Cancer']:<20}"
        print(row)

    # Performance improvement analysis
    accuracy_improvement = (
        fine_tuned_results["accuracy"] - baseline_results["accuracy"]
    ) * 100
    fn_reduction = (
        baseline_results["confusion_matrix"]["Actual Cancer"]["Predicted Non-Cancer"]
        - fine_tuned_results["confusion_matrix"]["Actual Cancer"][
            "Predicted Non-Cancer"
        ]
    )

    print("\n--- Performance Improvement Analysis ---")
    print(f"• Accuracy increased by {accuracy_improvement:.1f}% after fine-tuning")
    print(f"• Reduction in false negatives: {fn_reduction} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example", action="store_true", help="Evaluate on dummy abstract"
    )
    parser.add_argument("--train", action="store_true", help="Train and evaluate")
    parser.add_argument(
        "--full_extraction",
        action="store_true",
        help="Extract cancer from all abstracts",
    )
    args = parser.parse_args()

    classifierpipeline = CancerClassifierPipeline()
    if args.train:
        classifer_train_and_evaluate(classifierpipeline)

    extractionpipeline = CancerExtractionPipeline()
    if args.full_extraction:
        df = extract_cancer_from_abstract(extractionpipeline)
        df.to_csv("full_extraction_output.csv", index=False)

    if args.example:
        # Example prediction
        dummy_abstract = "This study investigates novel biomarkers for early detection of lung cancer in non-smokers. Patients with breast cancer and melanoma showed improved outcomes."
        print("\n--- Example Prediction & Extraction ---")
        print(f"Abstract: {dummy_abstract}")
        prediction = classifierpipeline.predict(dummy_abstract)
        print("Prediction Results:")
        print(json.dumps(prediction, indent=2))
        extractions = extractionpipeline.extract_diseases(dummy_abstract)
        extractions_cleaned = extractionpipeline.clean_diseases(extractions)
        detections = extractionpipeline.detect_cancer(extractions_cleaned)
        print("Detection Results::")
        print(detections)
