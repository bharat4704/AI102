"""
============================================================
AI-102 | Program 14 — OCR & Document Intelligence
Services: Azure AI Vision (Read API) + Document Intelligence
Skill   : Analyze images & documents
============================================================
Two approaches:
  1. Vision Read API  → general OCR from images
  2. Document Intelligence → structured document analysis
     (forms, invoices, receipts, ID cards, contracts)
============================================================
"""

import os
import time
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentAnalysisFeature,
)
from azure.core.credentials import AzureKeyCredential

VISION_ENDPOINT  = os.getenv("AZURE_VISION_ENDPOINT", "https://<vision>.cognitiveservices.azure.com/")
VISION_KEY       = os.getenv("AZURE_VISION_KEY", "<vision-key>")
DOC_ENDPOINT     = os.getenv("AZURE_DOCUMENT_ENDPOINT", "https://<doc-intel>.cognitiveservices.azure.com/")
DOC_KEY          = os.getenv("AZURE_DOCUMENT_KEY", "<doc-intel-key>")

def get_vision_client():
    return ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY)
    )

def get_doc_client():
    return DocumentIntelligenceClient(
        endpoint=DOC_ENDPOINT,
        credential=AzureKeyCredential(DOC_KEY)
    )

# ── 1. OCR with Vision Read API ───────────────────────────
def ocr_from_url(image_url: str) -> str:
    """
    Extract all text from an image URL using Vision Read API.
    Returns structured text with bounding polygons per word.
    """
    client = get_vision_client()

    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.READ]
    )

    print("\n" + "="*65)
    print("  OCR — VISION READ API")
    print("="*65)
    print(f"  URL: {image_url[:70]}")

    full_text = []
    if result.read:
        print(f"\n  Blocks: {len(result.read.blocks)}")
        for b_idx, block in enumerate(result.read.blocks):
            print(f"\n  Block {b_idx+1}:")
            for line in block.lines:
                print(f"    Line : '{line.text}'")
                full_text.append(line.text)
                for word in line.words:
                    poly = word.bounding_polygon
                    print(f"      Word: '{word.text}' conf:{word.confidence:.2f} "
                          f"poly:{[(p.x, p.y) for p in poly]}")
    else:
        print("  No text detected.")

    combined = " ".join(full_text)
    print(f"\n  Full text: {combined[:200]}")
    return combined

# ── 2. Document Intelligence — Prebuilt Models ────────────
def analyze_receipt(receipt_url: str) -> None:
    """
    Extract structured data from a receipt using prebuilt model.
    Extracts: MerchantName, Total, Tax, Items, Date, etc.
    """
    client = get_doc_client()

    poller = client.begin_analyze_document(
        model_id="prebuilt-receipt",
        analyze_request=AnalyzeDocumentRequest(url_source=receipt_url)
    )
    result = poller.result()

    print("\n" + "="*65)
    print("  DOCUMENT INTELLIGENCE — RECEIPT ANALYSIS")
    print("="*65)

    for doc in result.documents:
        print(f"  Document type: {doc.doc_type}")
        print(f"  Confidence  : {doc.confidence:.4f}\n")
        fields = doc.fields

        def get_field(name):
            f = fields.get(name)
            if f:
                return f.content or f.value_string or str(f.value_number or f.value_date or "")
            return "N/A"

        print(f"  Merchant   : {get_field('MerchantName')}")
        print(f"  Date       : {get_field('TransactionDate')}")
        print(f"  Total      : {get_field('Total')}")
        print(f"  Sub-total  : {get_field('Subtotal')}")
        print(f"  Tax        : {get_field('TotalTax')}")
        print(f"  Tip        : {get_field('Tip')}")

        items_field = fields.get("Items")
        if items_field and items_field.value_array:
            print(f"\n  Line Items:")
            for item in items_field.value_array:
                if item.value_object:
                    desc  = item.value_object.get("Description", {}).content or "?"
                    total = item.value_object.get("TotalPrice", {}).content or "?"
                    print(f"    • {desc:<30} {total}")

def analyze_invoice(invoice_url: str) -> None:
    """
    Extract structured data from an invoice.
    Extracts: Vendor, Customer, InvoiceId, DueDate, Items, Totals.
    """
    client = get_doc_client()

    poller = client.begin_analyze_document(
        model_id="prebuilt-invoice",
        analyze_request=AnalyzeDocumentRequest(url_source=invoice_url)
    )
    result = poller.result()

    print("\n" + "="*65)
    print("  DOCUMENT INTELLIGENCE — INVOICE ANALYSIS")
    print("="*65)

    for doc in result.documents:
        fields = doc.fields

        def get(name):
            f = fields.get(name)
            return f.content if f else "N/A"

        print(f"  Invoice ID  : {get('InvoiceId')}")
        print(f"  Invoice Date: {get('InvoiceDate')}")
        print(f"  Due Date    : {get('DueDate')}")
        print(f"  Vendor      : {get('VendorName')}")
        print(f"  Customer    : {get('CustomerName')}")
        print(f"  Amount Due  : {get('AmountDue')}")
        print(f"  Sub-total   : {get('SubTotal')}")
        print(f"  Tax         : {get('TotalTax')}")

def analyze_id_document(id_url: str) -> None:
    """
    Extract identity information from ID documents.
    Supports: passport, driver's licence, national ID.
    """
    client = get_doc_client()

    poller = client.begin_analyze_document(
        model_id="prebuilt-idDocument",
        analyze_request=AnalyzeDocumentRequest(url_source=id_url)
    )
    result = poller.result()

    print("\n" + "="*65)
    print("  DOCUMENT INTELLIGENCE — ID DOCUMENT")
    print("="*65)

    for doc in result.documents:
        fields = doc.fields
        def get(name):
            f = fields.get(name)
            return f.content if f else "N/A"

        print(f"  Document Type : {doc.doc_type}")
        print(f"  First Name    : {get('FirstName')}")
        print(f"  Last Name     : {get('LastName')}")
        print(f"  DOB           : {get('DateOfBirth')}")
        print(f"  Expiry        : {get('DateOfExpiration')}")
        print(f"  Document No   : {get('DocumentNumber')}")
        print(f"  Country       : {get('CountryRegion')}")

def analyze_general_document(doc_url: str) -> None:
    """
    Analyse any document — extracts key-value pairs and tables.
    Uses prebuilt-document model (no training required).
    """
    client = get_doc_client()

    poller = client.begin_analyze_document(
        model_id="prebuilt-document",
        analyze_request=AnalyzeDocumentRequest(url_source=doc_url)
    )
    result = poller.result()

    print("\n" + "="*65)
    print("  DOCUMENT INTELLIGENCE — GENERAL DOCUMENT")
    print("="*65)

    # Key-value pairs
    if result.key_value_pairs:
        print(f"\n  Key-Value Pairs ({len(result.key_value_pairs)}):")
        for kv in result.key_value_pairs[:15]:
            key   = kv.key.content if kv.key else "?"
            value = kv.value.content if kv.value else "?"
            print(f"    {key:<30} → {value}")

    # Tables
    if result.tables:
        print(f"\n  Tables ({len(result.tables)}):")
        for t_idx, table in enumerate(result.tables):
            print(f"    Table {t_idx+1}: {table.row_count}r × {table.column_count}c")
            for cell in table.cells[:10]:
                print(f"      [{cell.row_index},{cell.column_index}] '{cell.content}'")

    # Pages
    print(f"\n  Pages: {len(result.pages)}")
    for page in result.pages:
        print(f"    Page {page.page_number}: {page.width}x{page.height} {page.unit}")
        print(f"    Words: {len(page.words)} | Lines: {len(page.lines)}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # OCR test
    text_image_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
    ocr_from_url(text_image_url)

    # Document Intelligence tests
    receipt_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/rest-api/receipt.png"
    analyze_receipt(receipt_url)

    invoice_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/rest-api/invoice.pdf"
    analyze_invoice(invoice_url)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • Vision READ API = VisualFeatures.READ in analyze()")
    print("  • Document Intelligence uses DocumentIntelligenceClient")
    print("  • Prebuilt models: prebuilt-receipt, prebuilt-invoice,")
    print("    prebuilt-idDocument, prebuilt-document, prebuilt-layout")
    print("  • begin_analyze_document() is a long-running operation")
    print("  • doc.fields is a dict — use .get() safely")
    print("  • Tables: access via result.tables with row/col index")
    print("  • Custom models: train in Document Intelligence Studio")
    print("="*65 + "\n")
