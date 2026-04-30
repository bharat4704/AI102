"""
============================================================
AI-102 | Program 13 — Computer Vision — Image Analysis
Service : Azure AI Vision (v4.0)
Skill   : Analyze images with Azure AI Vision
============================================================
Features:
  • Caption generation (dense captions)
  • Object detection
  • Tag generation
  • People detection
  • Smart crops / region of interest
  • Brand & colour analysis
============================================================
"""

import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
VISION_KEY      = os.getenv("AZURE_VISION_KEY", "<your-vision-key>")

# Sample images for testing
IMG_URL_STREET  = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
IMG_URL_OFFICE  = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"

def get_client():
    return ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY)
    )

# ── 1. Full Image Analysis ────────────────────────────────
def analyze_image_url(image_url: str) -> None:
    """
    Comprehensive analysis of an image from URL.
    Requests multiple visual features in a single API call.
    """
    client = get_client()

    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.OBJECTS,
            VisualFeatures.TAGS,
            VisualFeatures.PEOPLE,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.READ,            # OCR
        ],
        smart_crops_aspect_ratios=[0.9, 1.33],  # Portrait and landscape
        gender_neutral_caption=True,            # Inclusive language
        language="en"
    )

    print("\n" + "="*65)
    print("  FULL IMAGE ANALYSIS")
    print("="*65)
    print(f"  URL: {image_url[:70]}")
    print(f"  Model version: {result.model_version}")
    print(f"  Image: {result.metadata.width}x{result.metadata.height}px\n")

    # Caption
    if result.caption:
        print(f"  📝 Caption: '{result.caption.text}'")
        print(f"     Confidence: {result.caption.confidence:.4f}")

    # Dense captions (multiple regions)
    if result.dense_captions:
        print(f"\n  📝 Dense Captions ({len(result.dense_captions.list)} regions):")
        for cap in result.dense_captions.list[:5]:
            bbox = cap.bounding_box
            print(f"     '{cap.text}' [{bbox.x},{bbox.y},{bbox.width}x{bbox.height}] "
                  f"conf:{cap.confidence:.2f}")

    # Tags
    if result.tags:
        print(f"\n  🏷️  Tags ({len(result.tags.list)} found):")
        top_tags = sorted(result.tags.list, key=lambda t: t.confidence, reverse=True)[:10]
        for tag in top_tags:
            bar = "█" * int(tag.confidence * 20)
            print(f"     {tag.name:<20} {bar} {tag.confidence:.3f}")

    # Objects
    if result.objects:
        print(f"\n  📦 Objects ({len(result.objects.list)} found):")
        for obj in result.objects.list:
            bbox = obj.bounding_box
            top_tag = obj.tags[0] if obj.tags else None
            if top_tag:
                print(f"     '{top_tag.name}' at [{bbox.x},{bbox.y}] "
                      f"size:{bbox.width}x{bbox.height} conf:{top_tag.confidence:.2f}")

    # People
    if result.people:
        print(f"\n  👤 People ({len(result.people.list)} detected):")
        for person in result.people.list:
            bbox = person.bounding_box
            print(f"     At [{bbox.x},{bbox.y}] size:{bbox.width}x{bbox.height} "
                  f"conf:{person.confidence:.2f}")

    # Smart crops
    if result.smart_crops:
        print(f"\n  ✂️  Smart Crops:")
        for crop in result.smart_crops.list:
            bbox = crop.bounding_box
            print(f"     Ratio {crop.aspect_ratio:.2f}: "
                  f"[{bbox.x},{bbox.y},{bbox.width}x{bbox.height}]")

    # Read (OCR)
    if result.read:
        print(f"\n  📖 OCR Text Detected:")
        for block in result.read.blocks:
            for line in block.lines:
                print(f"     '{line.text}'")

# ── 2. Analyse from Local File ────────────────────────────
def analyze_image_file(image_path: str) -> None:
    """
    Analyse an image from local file system.
    Reads file as bytes and sends to API.
    """
    client = get_client()

    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
        ]
    )

    print("\n" + "="*65)
    print("  IMAGE ANALYSIS — LOCAL FILE")
    print("="*65)
    print(f"  File: {image_path}")
    if result.caption:
        print(f"  Caption: '{result.caption.text}' ({result.caption.confidence:.2f})")
    if result.tags:
        print(f"  Tags: {', '.join(t.name for t in result.tags.list[:8])}")

# ── 3. Background Removal ─────────────────────────────────
def remove_background(image_url: str, output_path: str = "no_bg.png") -> None:
    """
    Remove image background using Azure AI Vision.
    Returns PNG with transparent background.
    """
    import requests

    url = f"{VISION_ENDPOINT}computervision/imageanalysis:segment"
    params = {"api-version": "2023-02-01-preview", "mode": "backgroundRemoval"}
    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/json"
    }
    body = {"url": image_url}

    print("\n" + "="*65)
    print("  BACKGROUND REMOVAL")
    print("="*65)

    response = requests.post(url, params=params, headers=headers, json=body)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"  ✅ Background removed. Saved to: {output_path}")
    else:
        print(f"  ❌ Error {response.status_code}: {response.text[:200]}")

# ── 4. Feature-Specific Analysis ─────────────────────────
def analyze_tags_only(image_url: str) -> list[str]:
    """Request only tags — faster and cheaper than full analysis."""
    client = get_client()
    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.TAGS]
    )
    tags = []
    if result.tags:
        tags = [f"{t.name}({t.confidence:.2f})" for t in result.tags.list]
        print(f"  Tags: {', '.join(tags[:10])}")
    return tags

def analyze_caption_only(image_url: str) -> str:
    """Request only caption — minimal tokens, quick response."""
    client = get_client()
    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.CAPTION],
        gender_neutral_caption=True
    )
    if result.caption:
        print(f"  Caption: '{result.caption.text}' ({result.caption.confidence:.2f})")
        return result.caption.text
    return ""

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    test_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"

    analyze_image_url(test_url)
    analyze_caption_only(test_url)
    analyze_tags_only(test_url)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • Azure AI Vision v4.0 uses ImageAnalysisClient")
    print("  • Older CV v3.2 uses ComputerVisionClient (legacy)")
    print("  • visual_features=[VisualFeatures.CAPTION, TAGS, OBJECTS, ...]")
    print("  • analyze_from_url() for URLs, analyze() for file bytes")
    print("  • gender_neutral_caption=True — always use in production")
    print("  • smart_crops_aspect_ratios=[] for thumbnail generation")
    print("  • VisualFeatures.READ = OCR (same service call)")
    print("  • Bounding boxes: {x, y, width, height} in pixels")
    print("="*65 + "\n")
