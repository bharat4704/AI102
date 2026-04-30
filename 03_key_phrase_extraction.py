import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY      = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")

def get_client():
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

def extract_key_phrases(documents: list[str]) -> None:
    """
    Extract the main talking points / key phrases from documents.
    Use cases: document indexing, search tagging, content summarisation.
    """
    client = get_client()
    results = client.extract_key_phrases(documents=documents)

    print("\n" + "="*55)
    print("  KEY PHRASE EXTRACTION RESULTS")
    print("="*55)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}:")
        print(f"  Text: \"{documents[i][:80]}\"")
        if not result.is_error:
            print(f"  Key Phrases ({len(result.key_phrases)} found):")
            for phrase in result.key_phrases:
                print(f"    • {phrase}")
        else:
            print(f"  ERROR: {result.error.code} — {result.error.message}")

# ── Batch Processing Example ───────────────────────────────
def batch_key_phrases(documents: list[str]) -> dict:
    """
    Returns a dictionary mapping document index to its key phrases.
    Demonstrates batch processing pattern for AI-102.
    """
    client = get_client()
    results = client.extract_key_phrases(documents=documents)
    output = {}
    for i, result in enumerate(results):
        if not result.is_error:
            output[i] = result.key_phrases
    return output

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    documents = [
        """
        Microsoft Azure AI Foundry provides a comprehensive platform for building
        intelligent applications. It includes services for natural language processing,
        computer vision, speech recognition, and decision making. Developers can use
        pre-built models or train custom models using their own data.
        """,
        """
        Cybersecurity is critical for protecting organisational assets. Threat detection,
        incident response, vulnerability assessment, and security monitoring are key
        components of a robust security posture. Zero trust architecture and multi-factor
        authentication help prevent unauthorised access.
        """,
        """
        Tanzania is experiencing rapid digital transformation. Mobile banking, e-commerce,
        and cloud adoption are growing significantly. Investment in digital infrastructure
        and cybersecurity training is essential for sustainable economic development.
        """,
    ]

    extract_key_phrases(documents)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • extract_key_phrases() returns list of string phrases per document")
    print("  • No confidence score — phrases are binary (present or not)")
    print("  • Useful for search indexing and document tagging pipelines")
    print("  • Supports batch processing — up to 1000 docs per call")
    print("  • Works best on longer documents with rich content")
    print("="*55 + "\n")
