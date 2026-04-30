

import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY      = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")

def get_client():
    return TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

# ── 1. Basic PII Detection ─────────────────────────────────
def detect_pii(documents: list[str]) -> None:
    """
    Detect PII entities and return automatically redacted text.
    doc.redacted_text replaces PII with ████ characters.
    """
    client = get_client()
    results = client.recognize_pii_entities(documents=documents, language="en")

    print("\n" + "="*65)
    print("  PII DETECTION & REDACTION")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}:")
        print(f"  Original : {documents[i]}")
        if not result.is_error:
            print(f"  Redacted : {result.redacted_text}")
            print(f"  PII Entities Found ({len(result.entities)}):")
            for entity in result.entities:
                print(f"    • [{entity.category}] '{entity.text}' "
                      f"(confidence: {entity.confidence_score:.2f})")
        else:
            print(f"  ERROR: {result.error.code} — {result.error.message}")

# ── 2. Domain-Specific PII ─────────────────────────────────
def detect_pii_by_domain(documents: list[str], domain: str = "phi") -> None:
    """
    domain='phi' — Protected Health Information (medical context)
    Adds medical-specific categories to detection.
    """
    client = get_client()
    # domain parameter adds PHI-specific categories
    results = client.recognize_pii_entities(
        documents=documents,
        language="en",
        domain_filter=domain      # 'phi' for medical records
    )

    print("\n" + "="*65)
    print(f"  PII DETECTION — DOMAIN: {domain.upper()}")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  Document {i+1}:")
        print(f"  Original : {documents[i]}")
        if not result.is_error:
            print(f"  Redacted : {result.redacted_text}")
            for entity in result.entities:
                print(f"    • [{entity.category}] '{entity.text}'")

# ── 3. Category Filtering ──────────────────────────────────
def detect_specific_pii(documents: list[str], categories: list[str]) -> None:
    """
    Filter PII detection to specific categories only.
    Reduces noise and focuses on what matters for your use case.
    """
    client = get_client()
    results = client.recognize_pii_entities(
        documents=documents,
        language="en",
        categories_filter=categories  # Only detect these PII types
    )

    print("\n" + "="*65)
    print(f"  PII — FILTERED TO: {categories}")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  Original : {documents[i]}")
        if not result.is_error:
            print(f"  Redacted : {result.redacted_text}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    general_pii_docs = [
        "My name is John Smith and my email is john.smith@example.com. Call me at +1-555-123-4567.",
        "Credit card: 4111-1111-1111-1111, expiry 12/27, CVV 123. Billing address: 123 Main St, Seattle WA 98101.",
        "SSN: 123-45-6789. Passport: AB1234567. IP address: 192.168.1.100.",
        "Please transfer $5,000 from account 1234567890, routing number 021000021.",
    ]

    phi_docs = [
        "Patient James Wilson (DOB: 05/14/1980) was admitted for Type 2 Diabetes treatment. MRN: 78934521.",
        "Dr. Sarah Jones prescribed Metformin 500mg. Insurance: BlueCross #BCX-987654.",
    ]

    detect_pii(general_pii_docs)
    detect_pii_by_domain(phi_docs, domain="phi")
    detect_specific_pii(
        general_pii_docs[:2],
        categories=["Email", "PhoneNumber"]   # Only find emails and phones
    )

    print("\n  KEY POINTS FOR AI-102:")
    print("  • recognize_pii_entities() — separate from recognize_entities()")
    print("  • doc.redacted_text = automatic redaction with ████")
    print("  • domain_filter='phi' for Protected Health Information")
    print("  • categories_filter=[] to limit which PII types to detect")
    print("  • PII detection is key for GDPR, HIPAA, data compliance scenarios")
    print("  • offset and length let you redact in original text manually")
    print("="*65 + "\n")
