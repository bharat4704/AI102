import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY      = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")

def get_client():
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

# ── 1. Basic Sentiment Analysis ────────────────────────────
def analyze_sentiment(documents: list[str]) -> None:
    """
    Document-level and sentence-level sentiment analysis.
    Sentiment values: positive | neutral | negative 
    """
    client = get_client()
    results = client.analyze_sentiment(documents=documents)

    print("\n" + "="*55)
    print("  SENTIMENT ANALYSIS")
    print("="*55)

    for i, result in enumerate(results):
        if not result.is_error:
            print(f"\n  Doc {i+1}: \"{documents[i][:55]}\"")
            print(f"  Overall Sentiment : {result.sentiment.upper()}")
            print(f"  Scores → Positive: {result.confidence_scores.positive:.2f} | "
                  f"Neutral: {result.confidence_scores.neutral:.2f} | "
                  f"Negative: {result.confidence_scores.negative:.2f}")

            print("  Sentence-level breakdown:")
            for j, sentence in enumerate(result.sentences):
                print(f"    [{j+1}] \"{sentence.text}\"")
                print(f"         → {sentence.sentiment} "
                      f"(+{sentence.confidence_scores.positive:.2f} "
                      f"~{sentence.confidence_scores.neutral:.2f} "
                      f"-{sentence.confidence_scores.negative:.2f})")

# ── 2. Opinion Mining (Aspect-Based Sentiment) ─────────────
def opinion_mining(documents: list[str]) -> None:
    """
    Aspect-based / opinion mining — extracts:
      Target   : the thing being discussed (e.g., 'food', 'service')
      Assessment: the opinion word (e.g., 'delicious', 'slow')
    Enabled via show_opinion_mining=True
    """
    client = get_client()
    results = client.analyze_sentiment(
        documents=documents,
        show_opinion_mining=True   # ← KEY parameter for AI-102
    )

    print("\n" + "="*55)
    print("  OPINION MINING (Aspect-Based Sentiment)")
    print("="*55)

    for i, result in enumerate(results):
        if not result.is_error:
            print(f"\n  Doc {i+1}: \"{documents[i][:55]}\"")
            print(f"  Overall: {result.sentiment}")
            for sentence in result.sentences:
                for opinion in sentence.mined_opinions:
                    target = opinion.target
                    print(f"\n    Target     : '{target.text}' → {target.sentiment}")
                    for assessment in opinion.assessments:
                        print(f"    Assessment : '{assessment.text}' → {assessment.sentiment} "
                              f"(negated: {assessment.is_negated})")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    general_docs = [
        "I absolutely love working with Azure AI services! The documentation is excellent.",
        "The service was okay, nothing special but it works.",
        "This is terrible. The API keeps failing and support is unresponsive.",
        "The food was delicious but the service was incredibly slow and the staff were rude.",
    ]

    opinion_docs = [
        "The hotel room was clean and spacious but the Wi-Fi was terrible.",
        "Azure Language service is fast and accurate but the pricing is confusing.",
        "The instructor was knowledgeable and patient. The course material was outdated though.",
    ]

    analyze_sentiment(general_docs)
    opinion_mining(opinion_docs)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • show_opinion_mining=True enables aspect-based sentiment")
    print("  • mined_opinions → target (noun) + assessments (adjectives)")
    print("  • assessment.is_negated handles 'not bad' → positive")
    print("  • Sentence-level scores differ from document-level scores")
    print("="*55 + "\n")
