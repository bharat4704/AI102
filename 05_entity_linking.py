import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY      = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")

def get_client():
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

def link_entities(documents: list[str]) -> None:
  
    client = get_client()
    results = client.recognize_linked_entities(documents=documents)

    print("\n" + "="*70)
    print("  ENTITY LINKING RESULTS")
    print("="*70)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}:")
        print(f"  Text: \"{documents[i][:80]}\"")
        if not result.is_error:
            for entity in result.entities:
                print(f"\n    Entity     : {entity.name}")
                print(f"    Wikipedia  : {entity.url}")
                print(f"    Data Source: {entity.data_source}")
                print(f"    Entity ID  : {entity.data_source_entity_id}")
                for match in entity.matches:
                    print(f"    Matched    : '{match.text}' "
                          f"(confidence: {match.confidence_score:.3f}, "
                          f"offset: {match.offset}, length: {match.length})")
        else:
            print(f"  ERROR: {result.error.code} — {result.error.message}")

# ── Disambiguation Demo ────────────────────────────────────
def demonstrate_disambiguation():
    """
    Shows how entity linking resolves ambiguous names.
    'Apple' can be the company or the fruit — linking resolves this.
    """
    client = get_client()
    ambiguous_docs = [
        "Apple released the new iPhone 16 today.",            # Apple = company
        "I ate a fresh apple from the garden this morning.",  # apple = fruit
        "Michael Jordan is considered the greatest basketball player ever.",
        "Jordan is a country located in the Middle East.",
    ]

    print("\n" + "="*70)
    print("  DISAMBIGUATION DEMO")
    print("="*70)

    results = client.recognize_linked_entities(documents=ambiguous_docs)
    for i, result in enumerate(results):
        print(f"\n  \"{ambiguous_docs[i]}\"")
        if not result.is_error:
            for entity in result.entities:
                print(f"  → '{entity.name}' linked to: {entity.url}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    documents = [
        "Satya Nadella became CEO of Microsoft in 2014 and has led the company's cloud transformation.",
        "Azure is Microsoft's cloud computing platform, competing with Amazon Web Services and Google Cloud.",
        "The Eiffel Tower in Paris was built by Gustave Eiffel and completed in 1889.",
        "Python is a popular programming language created by Guido van Rossum.",
    ]

    link_entities(documents)
    demonstrate_disambiguation()

    print("\n  KEY POINTS FOR AI-102:")
    print("  • recognize_linked_entities() — NOT recognize_entities()")
    print("  • Returns Wikipedia URLs and entity IDs for disambiguation")
    print("  • 'matches' = all occurrences of the entity in the document")
    print("  • Confidence score per match — not per entity globally")
    print("  • Key difference from NER: linking resolves WHICH 'Apple' is meant")
    print("="*70 + "\n")
