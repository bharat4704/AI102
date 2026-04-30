"""
============================================================
AI-102 | Program 08 — Custom Text Classification
Service : Azure AI Language (Language Studio)
Skill   : Build custom NLP solutions
============================================================
Two classification types:
  Single-label → one category per document
  Multi-label  → multiple categories per document
============================================================
PREREQUISITE:
  1. Go to language.cognitive.azure.com
  2. Create → Custom Text Classification project
  3. Upload and label documents
  4. Train and deploy model as 'production'
  5. Set PROJECT_NAME below
============================================================
"""

import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

ENDPOINT         = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY              = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")
PROJECT_NAME     = os.getenv("CLASSIFICATION_PROJECT", "<your-project-name>")
DEPLOYMENT_NAME  = os.getenv("CLASSIFICATION_DEPLOYMENT", "production")

def get_client():
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

# ── 1. Single-Label Classification ────────────────────────
def single_label_classify(documents: list[str]) -> None:
    """
    Assigns exactly ONE category to each document.
    Use when categories are mutually exclusive.
    e.g. News article → Sports / Politics / Technology / Business
    """
    client = get_client()

    # Long-running operation
    poller = client.begin_single_label_classify(
        documents=documents,
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME
    )
    results = poller.result()

    print("\n" + "="*60)
    print("  SINGLE-LABEL TEXT CLASSIFICATION")
    print("="*60)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}: \"{documents[i][:60]}\"")
        if not result.is_error:
            for classification in result.classifications:
                print(f"    Category   : {classification.category}")
                print(f"    Confidence : {classification.confidence_score:.4f}")
        else:
            print(f"  ERROR: {result.error.code} — {result.error.message}")

# ── 2. Multi-Label Classification ─────────────────────────
def multi_label_classify(documents: list[str]) -> None:
    """
    Assigns MULTIPLE categories to each document.
    Use when a document can belong to more than one category.
    e.g. Article can be both 'AI' and 'Cybersecurity'
    """
    client = get_client()

    poller = client.begin_multi_label_classify(
        documents=documents,
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME
    )
    results = poller.result()

    print("\n" + "="*60)
    print("  MULTI-LABEL TEXT CLASSIFICATION")
    print("="*60)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}: \"{documents[i][:60]}\"")
        if not result.is_error:
            # Sort by confidence descending
            sorted_classes = sorted(
                result.classifications,
                key=lambda c: c.confidence_score,
                reverse=True
            )
            for classification in sorted_classes:
                bar = "█" * int(classification.confidence_score * 20)
                print(f"    {classification.category:<20} {bar:<20} {classification.confidence_score:.4f}")
        else:
            print(f"  ERROR: {result.error.code} — {result.error.message}")

# ── 3. Classification with Threshold Filter ───────────────
def classify_with_threshold(documents: list[str], threshold: float = 0.70) -> None:
    """
    Only accept classifications above a confidence threshold.
    Prevents low-confidence labels from contaminating results.
    """
    client = get_client()
    poller = client.begin_multi_label_classify(
        documents=documents,
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME
    )
    results = poller.result()

    print("\n" + "="*60)
    print(f"  CLASSIFICATION — THRESHOLD: {threshold}")
    print("="*60)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}: \"{documents[i][:60]}\"")
        if not result.is_error:
            accepted = [c for c in result.classifications if c.confidence_score >= threshold]
            rejected = [c for c in result.classifications if c.confidence_score < threshold]
            for c in accepted:
                print(f"    ✅ {c.category} ({c.confidence_score:.2f})")
            for c in rejected:
                print(f"    ❌ {c.category} ({c.confidence_score:.2f}) — below threshold")

# ── Language Studio Setup Guide ────────────────────────────
def print_setup_guide():
    print("\n" + "="*60)
    print("  LANGUAGE STUDIO SETUP GUIDE")
    print("="*60)
    steps = [
        "1. Go to language.cognitive.azure.com",
        "2. Click 'Create new' → 'Custom text classification'",
        "3. Choose Single-label or Multi-label",
        "4. Create storage account and container for training data",
        "5. Upload documents (.txt files, one per sample)",
        "6. Label each document with categories in the portal",
        "7. Minimum: 10 documents per class recommended",
        "8. Click 'Train' → choose Quick or Standard training",
        "9. Evaluate model performance (Precision, Recall, F1)",
        "10. Deploy to production endpoint",
        "11. Update PROJECT_NAME and DEPLOYMENT_NAME in .env",
    ]
    for step in steps:
        print(f"  {step}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample documents — adjust to your trained categories
    documents = [
        "Azure Sentinel detected a SQL injection attack attempt on the web application firewall.",
        "Microsoft announced record quarterly earnings with cloud revenue up 29% year over year.",
        "The neural network model achieved 94% accuracy on the image classification benchmark.",
        "The cybersecurity training covered phishing, ransomware, and social engineering attacks.",
    ]

    print_setup_guide()
    single_label_classify(documents)
    multi_label_classify(documents)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • Custom classification requires Language Studio training first")
    print("  • Single-label: begin_single_label_classify()")
    print("  • Multi-label:  begin_multi_label_classify()")
    print("  • Both are long-running — use begin_ + .result()")
    print("  • Evaluate with Precision, Recall, F1 in Language Studio")
    print("  • Deploy to named deployment (e.g., 'production', 'staging')")
    print("="*60 + "\n")
