import os
import re
import evaluate
import argparse

def parse_labeled_predictions(file_path):
    predictions = []
    references = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"([LP])\s*\[\d+\]\s*\t(.+)", line.strip())
            if match:
                tag, text = match.groups()
                if tag == "L":
                    references.append([text])
                elif tag == "P":
                    predictions.append(text)

    return predictions, references

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_name", type=str, default="chrf", help="Metric to compute (e.g., chrf, sacrebleu)")
    parser.add_argument("--predictions_labels", type=str, required=True, help="Path to the labeled predictions file")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_labels):
        raise FileNotFoundError(f"❌ Labeled predictions file not found: {args.predictions_labels}")

    print(f"✅ Using labeled predictions file: {args.predictions_labels}")
    predictions, references = parse_labeled_predictions(args.predictions_labels)

    metric = evaluate.load(args.metric_name)
    results = metric.compute(predictions=predictions, references=references)

    print(f"{args.metric_name} score: {results['score']:.2f}")

if __name__ == "__main__":
    main()
