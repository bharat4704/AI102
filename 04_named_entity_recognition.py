import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY      = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")

def get_client():
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

def recognize_entities(documents: list[str]) -> None:
    """
    Identify and categorise named entities in text.
    Each entity has: text, category, subcategory, confidence_score, offset, length
    """
    client = get_client()
    results = client.recognize_entities(documents=documents)

    print("\n" + "="*65)
    print("  NAMED ENTITY RECOGNITION (NER) RESULTS")
    print("="*65)
    print(f"  {'ENTITY TEXT':<25} {'CATEGORY':<20} {'SUBCATEGORY':<18} {'CONF'}")
    print(f"  {'-'*24} {'-'*19} {'-'*17} {'-'*6}")

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}: \"{documents[i][:70]}\"")
        if not result.is_error:
            if result.entities:
                for entity in result.entities:
                    subcat = entity.subcategory if entity.subcategory else "—"
                    print(f"  {entity.text:<25} {entity.category:<20} {subcat:<18} {entity.confidence_score:.2f}")
            else:
                print("  No entities found.")
        else:
            print(f"  ERROR: {result.error.code} — {result.error.message}")

def extract_by_category(documents: list[str], category: str) -> list[str]:
    """
    Filter entities by a specific category.
    Example: extract only Person entities from all documents.
    """
    client = get_client()
    results = client.recognize_entities(documents=documents)
    found = []
    for result in results:
        if not result.is_error:
            for entity in result.entities:
                if entity.category == category and entity.text not in found:
                    found.append(entity.text)
    return found

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    documents = [
        "Satya Nadella, CEO of Microsoft, announced Azure AI Foundry at the Microsoft Build conference in Seattle on May 21, 2024.",
        "The cybersecurity training was held in Dar es Salaam, Tanzania. Contact us at training@cybersec.co.tz or call +255-712-000-000.",
        "Azure AI services include Computer Vision, Language Service, and Speech Studio. Visit https://azure.microsoft.com for pricing.",
        "The project budget is $50,000 and must be completed by December 31, 2025. The team consists of 12 engineers.",
    ]

    recognize_entities(documents)

    # Filter example
    print("\n  FILTER EXAMPLE — Person entities only:")
    persons = extract_by_category(documents, "Person")
    for p in persons:
        print(f"    • {p}")

    print("\n  KEY POINTS FOR AI-102:")
    print("  • Each entity: text, category, subcategory, confidence_score, offset, length")
    print("  • offset/length = character position in original string")
    print("  • Entity Linking is DIFFERENT — links to Wikipedia (Program 05)")
    print("  • NER categories tested: Person, Location, Organisation, DateTime, Quantity")
    print("  • Email, URL, PhoneNumber, IPAddress are also returned as entities")
    print("="*65 + "\n")
