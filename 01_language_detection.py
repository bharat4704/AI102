import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# ── Configuration ──────────────────────────────────────────
ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY      = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")

def get_client():
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

# ── Core Function ──────────────────────────────────────────
def detect_language(documents: list[str]) -> None:
    """
    Detect the language of each document.
    Returns: language name, ISO 639-1 code, confidence score.
    """
    client = get_client()
    results = client.detect_language(documents=documents)

    print("\n" + "="*55)
    print("  LANGUAGE DETECTION RESULTS")
    print("="*55)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}: \"{documents[i][:60]}\"")
        if not result.is_error:
            lang = result.primary_language
            print(f"    Language   : {lang.name}")
            print(f"    ISO Code   : {lang.iso6391_name}")
            print(f"    Confidence : {lang.confidence_score:.4f}")
        else:
            print(f"    ERROR: {result.error.code} — {result.error.message}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    sample_documents = [
        "Hello, how are you today?",                          # English
        "Bonjour, comment allez-vous aujourd'hui?",           # French
        "Hola, ¿cómo estás hoy?",                            # Spanish
        "नमस्ते, आज आप कैसे हैं?",                          # Hindi
        "مرحبا، كيف حالك اليوم؟",                           # Arabic
        "Habari yako leo?",                                   # Swahili
        "今日はお元気ですか？",                               # Japanese
        "Guten Tag, wie geht es Ihnen?",                      # German
    ]

    detect_language(sample_documents)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • detect_language() accepts up to 1000 documents per call")
    print("  • Returns primary_language with name, iso6391_name, confidence_score")
    print("  • Confidence score of 0 means language could not be detected")
    print("  • Use 'unknown' documents to test error handling")
    print("="*55 + "\n")
