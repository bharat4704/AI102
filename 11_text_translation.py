"""
============================================================
AI-102 | Program 11 — Text Translation
Service : Azure AI Translator
Skill   : Process and translate text
============================================================
Features:
  • Translate to multiple languages in one call
  • Auto language detection
  • Transliteration (script conversion)
  • Dictionary lookup & examples
  • Language enumeration
============================================================
"""

import os
from azure.ai.translation.text import TextTranslationClient
from azure.ai.translation.text.models import InputTextItem
from azure.core.credentials import AzureKeyCredential

TRANSLATOR_KEY    = os.getenv("AZURE_TRANSLATOR_KEY", "<your-translator-key>")
TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "eastus")
TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")

def get_client():
    credential = AzureKeyCredential(TRANSLATOR_KEY)
    return TextTranslationClient(
        credential=credential,
        region=TRANSLATOR_REGION
    )

# ── 1. Basic Translation ───────────────────────────────────
def translate_text(texts: list[str], target_languages: list[str]) -> None:
    """
    Translate text into one or more target languages.
    Auto-detects source language if not specified.
    """
    client = get_client()
    input_items = [InputTextItem(text=t) for t in texts]

    results = client.translate(
        body=input_items,
        to_language=target_languages
    )

    print("\n" + "="*65)
    print("  TEXT TRANSLATION")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  Source: '{texts[i]}'")
        if result.detected_language:
            print(f"  Detected: {result.detected_language.language} "
                  f"(confidence: {result.detected_language.score:.2f})")
        for translation in result.translations:
            print(f"  [{translation.to.upper()}]: {translation.text}")

# ── 2. Translate with Explicit Source Language ─────────────
def translate_with_source(texts: list[str], from_lang: str, to_langs: list[str]) -> None:
    """
    Specify the source language explicitly.
    Faster than auto-detection when source is known.
    """
    client = get_client()
    input_items = [InputTextItem(text=t) for t in texts]

    results = client.translate(
        body=input_items,
        from_language=from_lang,
        to_language=to_langs
    )

    print("\n" + "="*65)
    print(f"  TRANSLATION FROM: {from_lang.upper()}")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  [{from_lang}]: '{texts[i]}'")
        for translation in result.translations:
            print(f"  [{translation.to}]: {translation.text}")

# ── 3. Transliteration ────────────────────────────────────
def transliterate_text(texts: list[str], language: str,
                        from_script: str, to_script: str) -> None:
    """
    Convert text between different scripts without translating meaning.
    e.g. Hindi Devanagari → Latin (romanisation)
    e.g. Japanese Kanji → Hiragana
    """
    client = get_client()
    input_items = [InputTextItem(text=t) for t in texts]

    results = client.transliterate(
        body=input_items,
        language=language,
        from_script=from_script,
        to_script=to_script
    )

    print("\n" + "="*65)
    print(f"  TRANSLITERATION: {from_script} → {to_script}")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  Original ({from_script}): {texts[i]}")
        print(f"  Romanised ({to_script}) : {result.text}")
        print(f"  Script used            : {result.script}")

# ── 4. Dictionary Lookup ───────────────────────────────────
def dictionary_lookup(words: list[str], from_lang: str, to_lang: str) -> None:
    """
    Look up alternative translations for single words.
    Returns part of speech, back-translations, and confidence.
    """
    client = get_client()
    input_items = [InputTextItem(text=w) for w in words]

    results = client.lookup_dictionary_entries(
        body=input_items,
        from_language=from_lang,
        to_language=to_lang
    )

    print("\n" + "="*65)
    print(f"  DICTIONARY LOOKUP: {from_lang.upper()} → {to_lang.upper()}")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  Word: '{words[i]}'")
        for translation in result.translations:
            print(f"  Translation : {translation.display_target} "
                  f"[{translation.pos_tag}] "
                  f"(confidence: {translation.confidence:.2f})")
            if translation.back_translations:
                backs = [b.display_text for b in translation.back_translations[:3]]
                print(f"  Back-trans  : {', '.join(backs)}")

# ── 5. Get Supported Languages ────────────────────────────
def list_supported_languages() -> None:
    """
    List all languages supported by the translation service.
    Covers: translation, transliteration, dictionary.
    """
    client = get_client()
    result = client.get_supported_languages()

    print("\n" + "="*65)
    print("  SUPPORTED LANGUAGES")
    print("="*65)

    if result.translation:
        lang_list = list(result.translation.items())[:20]  # Show first 20
        print(f"\n  Translation languages (showing 20 of {len(result.translation)}):")
        for code, lang in lang_list:
            print(f"    {code:<8} {lang.name:<25} (native: {lang.native_name})")

    if result.transliteration:
        print(f"\n  Transliteration languages: {len(result.transliteration)}")

    if result.dictionary:
        print(f"  Dictionary languages: {len(result.dictionary)}")

# ── 6. Detect Language via Translator ────────────────────
def detect_language_translator(texts: list[str]) -> None:
    """
    Use Translator service to detect language.
    Translator's detection is separate from Language service's.
    """
    client = get_client()
    input_items = [InputTextItem(text=t) for t in texts]

    results = client.detect_language(body=input_items)

    print("\n" + "="*65)
    print("  LANGUAGE DETECTION (via Translator)")
    print("="*65)

    for i, result in enumerate(results):
        print(f"\n  '{texts[i][:60]}'")
        print(f"  Language  : {result.language}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Is Trans  : {result.is_translation_supported}")
        print(f"  Is Trans  : {result.is_transliteration_supported}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    english_texts = [
        "Cybersecurity is critical for protecting digital assets.",
        "Artificial intelligence is transforming every industry.",
        "Welcome to the Azure AI Foundry training programme.",
    ]

    hindi_texts = [
        "नमस्ते, आज का प्रशिक्षण कार्यक्रम शुरू होता है।",
        "साइबर सुरक्षा बहुत महत्वपूर्ण है।",
    ]

    # 1. Translate English to multiple languages
    translate_text(
        texts=english_texts[:2],
        target_languages=["fr", "es", "hi", "sw", "ar", "zh-Hans"]
    )

    # 2. Explicit source language
    translate_with_source(
        texts=hindi_texts,
        from_lang="hi",
        to_langs=["en", "sw", "fr"]
    )

    # 3. Transliterate Hindi to Latin
    transliterate_text(
        texts=hindi_texts,
        language="hi",
        from_script="Deva",   # Devanagari
        to_script="Latn"      # Latin/Roman
    )

    # 4. Dictionary lookup
    dictionary_lookup(
        words=["security", "cloud", "intelligence"],
        from_lang="en",
        to_lang="fr"
    )

    # 5. List languages
    list_supported_languages()

    # 6. Detect via Translator
    detect_language_translator([
        "Hello world",
        "Bonjour le monde",
        "Hujambo dunia",   # Swahili
    ])

    print("\n  KEY POINTS FOR AI-102:")
    print("  • Azure Translator is SEPARATE service from Language service")
    print("  • TextTranslationClient needs KEY + REGION (not just endpoint)")
    print("  • to_language=[] — pass a list for multi-language output")
    print("  • from_language omitted = auto-detection")
    print("  • Transliteration = script conversion, NOT meaning translation")
    print("  • Dictionary lookup works only for single words/phrases")
    print("  • Language codes: 'en', 'fr', 'hi', 'sw', 'zh-Hans', 'ar'")
    print("="*65 + "\n")
