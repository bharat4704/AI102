"""
============================================================
AI-102 | Program 20 — Responsible AI Principles
Service : Cross-cutting concern for ALL Azure AI services
Skill   : Responsible AI and governance
============================================================
Microsoft Responsible AI Principles:
  1. Fairness        — Equal treatment across groups
  2. Reliability     — Safe and consistent behaviour
  3. Privacy         — Data protection and security
  4. Inclusiveness   — Empowering everyone
  5. Transparency    — Understandable decisions
  6. Accountability  — Human oversight and control
============================================================
"""

import os
import json
from azure.ai.textanalytics import TextAnalyticsClient
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential

LANGUAGE_ENDPOINT        = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your>.cognitiveservices.azure.com/")
LANGUAGE_KEY             = os.getenv("AZURE_LANGUAGE_KEY", "<key>")
CONTENT_SAFETY_ENDPOINT  = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT", "https://<your>.cognitiveservices.azure.com/")
CONTENT_SAFETY_KEY       = os.getenv("AZURE_CONTENT_SAFETY_KEY", "<key>")

# ── 1. Fairness — Bias Detection ──────────────────────────
def check_for_bias(texts: list[str]) -> None:
    """
    Demonstrate fairness checks on AI outputs.
    Compare sentiment across different demographic groups
    to identify potential bias in analysis.
    """
    client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    print("\n" + "="*65)
    print("  PRINCIPLE 1: FAIRNESS — Bias Detection")
    print("="*65)
    print("  Testing if similar contexts yield similar results")
    print("  across different demographic groups:\n")

    results = client.analyze_sentiment(documents=texts)

    for i, (text, result) in enumerate(zip(texts, results)):
        if not result.is_error:
            print(f"  [{i+1}] '{text}'")
            print(f"       Sentiment: {result.sentiment} "
                  f"(+{result.confidence_scores.positive:.2f} "
                  f"-{result.confidence_scores.negative:.2f})")

    print("\n  ACTION: Compare results — if similar texts about")
    print("  different groups yield different sentiment scores,")
    print("  investigate for potential model bias.")

# ── 2. Reliability — Error Handling & Fallback ────────────
def reliable_analysis(text: str) -> dict:
    """
    Implement robust error handling for production AI systems.
    Includes retry logic, graceful degradation, and logging.
    """
    import time

    client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    print("\n" + "="*65)
    print("  PRINCIPLE 2: RELIABILITY — Robust Error Handling")
    print("="*65)

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            results = client.analyze_sentiment(documents=[text])
            result = results[0]

            if result.is_error:
                print(f"  API Error: {result.error.code} — {result.error.message}")
                return {"error": result.error.message, "fallback": True}

            output = {
                "sentiment": result.sentiment,
                "confidence": max(
                    result.confidence_scores.positive,
                    result.confidence_scores.neutral,
                    result.confidence_scores.negative
                ),
                "reliable": True,
                "attempts": attempt + 1
            }
            print(f"  ✅ Analysis succeeded on attempt {attempt + 1}")
            print(f"  Result: {output}")
            return output

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2   # Exponential backoff

    # Graceful degradation
    print("  All attempts failed — returning safe fallback response")
    return {"error": "Service unavailable", "fallback": True, "reliable": False}

# ── 3. Privacy — PII Handling Pipeline ───────────────────
def privacy_safe_pipeline(text: str) -> str:
    """
    Before analysing text, automatically detect and redact PII.
    Implements privacy-by-design principle.
    """
    client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    print("\n" + "="*65)
    print("  PRINCIPLE 3: PRIVACY — PII-Safe Analysis Pipeline")
    print("="*65)
    print(f"  Original : {text}")

    # Step 1: Detect and redact PII first
    pii_results = client.recognize_pii_entities(documents=[text], language="en")
    pii_result  = pii_results[0]

    if not pii_result.is_error:
        redacted_text = pii_result.redacted_text
        print(f"  Redacted : {redacted_text}")
        print(f"  PII found: {len(pii_result.entities)} items")
        for entity in pii_result.entities:
            print(f"    • [{entity.category}] redacted '{entity.text}'")

        # Step 2: Analyse the REDACTED text only
        sentiment_results = client.analyze_sentiment(documents=[redacted_text])
        sentiment = sentiment_results[0]
        if not sentiment.is_error:
            print(f"  Analysis on redacted text: {sentiment.sentiment}")
            print(f"  ✅ Privacy protected — PII never sent to sentiment API")

        return redacted_text
    return text

# ── 4. Transparency — Explain AI Decisions ───────────────
def transparent_decision(text: str) -> None:
    """
    Make AI decisions explainable with scores and reasoning.
    Never give a binary output without confidence/evidence.
    """
    client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    print("\n" + "="*65)
    print("  PRINCIPLE 4: TRANSPARENCY — Explainable Output")
    print("="*65)

    results = client.analyze_sentiment(documents=[text], show_opinion_mining=True)
    result  = results[0]

    if not result.is_error:
        print(f"  Input   : '{text}'")
        print(f"\n  DECISION: Sentiment is {result.sentiment.upper()}")
        print(f"\n  EVIDENCE (why we made this decision):")
        print(f"    Positive score: {result.confidence_scores.positive:.4f}")
        print(f"    Neutral score : {result.confidence_scores.neutral:.4f}")
        print(f"    Negative score: {result.confidence_scores.negative:.4f}")

        print(f"\n  SENTENCE-LEVEL BREAKDOWN:")
        for i, sentence in enumerate(result.sentences):
            print(f"    [{i+1}] '{sentence.text}'")
            print(f"         → {sentence.sentiment}")
            for opinion in sentence.mined_opinions:
                target = opinion.target
                print(f"         Target '{target.text}': {target.sentiment}")
                for assessment in opinion.assessments:
                    print(f"           Assessment '{assessment.text}': {assessment.sentiment}")

        print(f"\n  ℹ️  Users should understand why the AI reached this conclusion.")
        print(f"  ℹ️  High confidence (>0.9) → reliable; Low (<0.6) → review manually.")

# ── 5. Accountability — Human-in-the-Loop ────────────────
def human_in_the_loop_workflow(texts: list[str], confidence_threshold: float = 0.85) -> None:
    """
    Route low-confidence AI decisions to human review.
    High confidence → automated decision.
    Low confidence → escalate to human reviewer.
    """
    client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    print("\n" + "="*65)
    print("  PRINCIPLE 5: ACCOUNTABILITY — Human-in-the-Loop")
    print(f"  Confidence threshold: {confidence_threshold}")
    print("="*65)

    results = client.analyze_sentiment(documents=texts)
    automated, review_queue = [], []

    for i, (text, result) in enumerate(zip(texts, results)):
        if not result.is_error:
            top_conf = max(
                result.confidence_scores.positive,
                result.confidence_scores.neutral,
                result.confidence_scores.negative
            )
            decision = {
                "text": text,
                "sentiment": result.sentiment,
                "confidence": top_conf
            }
            if top_conf >= confidence_threshold:
                automated.append(decision)
                print(f"  ✅ AUTO  [{top_conf:.2f}] '{text[:50]}' → {result.sentiment}")
            else:
                review_queue.append(decision)
                print(f"  👤 HUMAN [{top_conf:.2f}] '{text[:50]}' → needs review")

    print(f"\n  Summary:")
    print(f"    Automated decisions : {len(automated)}")
    print(f"    Sent to human review: {len(review_queue)}")
    print(f"    ℹ️  Human review queue should be monitored and actioned regularly.")

# ── 6. Inclusiveness — Multi-Language Support ─────────────
def inclusive_multilingual_analysis(texts_by_language: dict) -> None:
    """
    Ensure AI services work equitably across languages.
    Demonstrate language detection + analysis pipeline
    that works regardless of input language.
    """
    client = TextAnalyticsClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    print("\n" + "="*65)
    print("  PRINCIPLE 6: INCLUSIVENESS — Multi-Language Pipeline")
    print("="*65)

    all_texts = list(texts_by_language.values())

    # Step 1: Detect language
    lang_results = client.detect_language(documents=all_texts)

    # Step 2: Analyse sentiment per language
    for i, (expected_lang, text) in enumerate(texts_by_language.items()):
        detected = lang_results[i].primary_language
        print(f"\n  Language [{expected_lang}]: '{text[:60]}'")
        print(f"  Detected: {detected.name} ({detected.iso6391_name}) "
              f"conf:{detected.confidence_score:.2f}")

        # Analyse with detected language
        sentiment_results = client.analyze_sentiment(
            documents=[text],
            language=detected.iso6391_name
        )
        sent = sentiment_results[0]
        if not sent.is_error:
            print(f"  Sentiment: {sent.sentiment} "
                  f"(+{sent.confidence_scores.positive:.2f} "
                  f"-{sent.confidence_scores.negative:.2f})")

# ── 7. Responsible AI Checklist ───────────────────────────
def print_rai_checklist():
    print("\n" + "="*65)
    print("  RESPONSIBLE AI CHECKLIST — AI-102 EXAM")
    print("="*65)

    checklist = {
        "Fairness": [
            "Test AI outputs across diverse demographic groups",
            "Monitor for bias in training data",
            "Use Azure Responsible AI dashboard in AML",
        ],
        "Reliability & Safety": [
            "Implement retry logic and exponential backoff",
            "Define fallback responses for failed API calls",
            "Apply Content Safety filters in production",
            "Use confidence thresholds for automated decisions",
        ],
        "Privacy & Security": [
            "Detect and redact PII before sending to AI services",
            "Use Managed Identity instead of hardcoded keys",
            "Store keys in Azure Key Vault",
            "Apply data minimisation principle",
        ],
        "Inclusiveness": [
            "Support multiple languages where users require",
            "Use gender_neutral_caption=True in Vision",
            "Test with diverse user groups",
        ],
        "Transparency": [
            "Always return confidence scores with AI decisions",
            "Log all AI decisions with reasoning",
            "Provide model cards for deployed models",
            "Communicate AI capabilities and limitations to users",
        ],
        "Accountability": [
            "Implement human-in-the-loop for high-stakes decisions",
            "Set clear escalation paths for edge cases",
            "Assign AI system owners and reviewers",
            "Conduct regular bias and performance audits",
        ],
    }

    for principle, items in checklist.items():
        print(f"\n  {principle}:")
        for item in items:
            print(f"    ☐ {item}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Fairness check
    demographic_texts = [
        "The engineer solved the complex problem brilliantly.",
        "The nurse provided excellent patient care today.",
        "The teacher inspired students with creative lessons.",
    ]
    check_for_bias(demographic_texts)

    # 2. Reliability
    reliable_analysis("Azure AI services provide robust NLP capabilities.")

    # 3. Privacy pipeline
    privacy_safe_pipeline(
        "John Smith, email john@test.com, called about his account #12345."
    )

    # 4. Transparency
    transparent_decision(
        "The service was excellent but the wait time was too long."
    )

    # 5. Human-in-the-loop
    mixed_confidence_texts = [
        "I absolutely love this product! It exceeded all my expectations.",
        "The item was fine I guess.",
        "Worst purchase ever. Completely broken on arrival.",
        "It was okay.",
    ]
    human_in_the_loop_workflow(mixed_confidence_texts, confidence_threshold=0.85)

    # 6. Inclusiveness
    multilingual = {
        "English": "The training was excellent and very informative.",
        "French": "La formation était excellente et très informative.",
        "Hindi": "प्रशिक्षण उत्कृष्ट और बहुत जानकारीपूर्ण था।",
        "Swahili": "Mafunzo yalikuwa bora na ya kuelimisha sana.",
    }
    inclusive_multilingual_analysis(multilingual)

    # 7. Checklist
    print_rai_checklist()

    print("\n  KEY POINTS FOR AI-102:")
    print("  • 6 RAI principles: Fairness, Reliability, Privacy,")
    print("    Inclusiveness, Transparency, Accountability")
    print("  • Content Safety Service = primary RAI enforcement tool")
    print("  • PII detection before analysis = Privacy by Design")
    print("  • Confidence thresholds → human-in-the-loop routing")
    print("  • gender_neutral_caption=True in all Vision calls")
    print("  • Azure AI Foundry includes RAI evaluation tools")
    print("  • Limited Access features: Face emotions, Custom Neural Voice")
    print("  • Always log AI decisions with timestamps for accountability")
    print("="*65 + "\n")
