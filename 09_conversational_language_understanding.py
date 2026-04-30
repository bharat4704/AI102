"""
============================================================
AI-102 | Program 09 — Conversational Language Understanding
Service : Azure AI Language (CLU)
Skill   : Build custom NLP solutions
============================================================
CLU replaces LUIS (retired Sept 2025).
Purpose: Detect user INTENT and extract ENTITIES from
         natural language utterances in conversational apps.

Key Concepts:
  Utterance : what the user says ("Book a flight to Paris")
  Intent    : what they want (BookFlight)
  Entity    : specific values (Paris = destination)
============================================================
"""

import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.ai.language.conversations.models import (
    CustomConversationalTaskParameters,
    TextConversationItem,
    AnalyzeConversationInput,
)

ENDPOINT         = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY              = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")
CLU_PROJECT      = os.getenv("CLU_PROJECT_NAME", "<your-clu-project>")
CLU_DEPLOYMENT   = os.getenv("CLU_DEPLOYMENT_NAME", "production")

def get_client():
    return ConversationAnalysisClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY)
    )

# ── 1. Analyze Single Utterance ───────────────────────────
def analyze_utterance(user_text: str) -> dict:
    """
    Analyze a single user utterance to detect intent and entities.
    Returns structured prediction with top intent and entities.
    """
    client = get_client()

    task = AnalyzeConversationInput(
        conversation_item=TextConversationItem(
            participant_id="user1",
            id="1",
            text=user_text,
            language="en"
        ),
        parameters=CustomConversationalTaskParameters(
            project_name=CLU_PROJECT,
            deployment_name=CLU_DEPLOYMENT
        ),
        kind="Conversation"
    )

    result = client.analyze_conversation(task=task)
    prediction = result.results.prediction

    print("\n" + "="*60)
    print("  CLU — INTENT & ENTITY DETECTION")
    print("="*60)
    print(f"\n  Utterance  : '{user_text}'")
    print(f"  Top Intent : {prediction.top_intent}")
    print(f"  Confidence : {prediction.intents[0].confidence:.4f}")

    print(f"\n  All Intents:")
    for intent in prediction.intents:
        bar = "█" * int(intent.confidence * 30)
        print(f"    {intent.category:<25} {bar} {intent.confidence:.4f}")

    print(f"\n  Entities ({len(prediction.entities)} found):")
    for entity in prediction.entities:
        print(f"    • [{entity.category}] '{entity.text}' "
              f"(confidence: {entity.confidence_score:.2f})")
        if entity.extra_information:
            for info in entity.extra_information:
                print(f"      Extra: {info}")

    return {
        "intent": prediction.top_intent,
        "confidence": prediction.intents[0].confidence,
        "entities": [(e.category, e.text) for e in prediction.entities]
    }

# ── 2. Multi-Turn Conversation ────────────────────────────
def multi_turn_conversation(utterances: list[str]) -> None:
    """
    Simulate a multi-turn conversation where context carries forward.
    Each utterance gets a unique ID for tracking.
    """
    client = get_client()

    print("\n" + "="*60)
    print("  MULTI-TURN CONVERSATION SIMULATION")
    print("="*60)

    for i, utterance in enumerate(utterances):
        task = AnalyzeConversationInput(
            conversation_item=TextConversationItem(
                participant_id="user1",
                id=str(i + 1),
                text=utterance,
                language="en"
            ),
            parameters=CustomConversationalTaskParameters(
                project_name=CLU_PROJECT,
                deployment_name=CLU_DEPLOYMENT
            ),
            kind="Conversation"
        )

        result = client.analyze_conversation(task=task)
        prediction = result.results.prediction

        print(f"\n  Turn {i+1}: '{utterance}'")
        print(f"    Intent  : {prediction.top_intent} ({prediction.intents[0].confidence:.2f})")
        if prediction.entities:
            for entity in prediction.entities:
                print(f"    Entity  : [{entity.category}] '{entity.text}'")

# ── 3. Batch Utterance Analysis ───────────────────────────
def batch_analyze(utterances: list[str]) -> list[dict]:
    """
    Process multiple utterances and return structured results.
    Useful for testing intent coverage of your CLU model.
    """
    client = get_client()
    results_list = []

    print("\n" + "="*60)
    print("  BATCH UTTERANCE ANALYSIS")
    print("="*60)
    print(f"\n  {'UTTERANCE':<40} {'INTENT':<20} {'CONF'}")
    print(f"  {'-'*39} {'-'*19} {'-'*6}")

    for i, utterance in enumerate(utterances):
        task = AnalyzeConversationInput(
            conversation_item=TextConversationItem(
                participant_id="user1",
                id=str(i),
                text=utterance,
                language="en"
            ),
            parameters=CustomConversationalTaskParameters(
                project_name=CLU_PROJECT,
                deployment_name=CLU_DEPLOYMENT
            ),
            kind="Conversation"
        )
        result = client.analyze_conversation(task=task)
        pred = result.results.prediction
        print(f"  {utterance[:39]:<40} {pred.top_intent:<20} {pred.intents[0].confidence:.4f}")
        results_list.append({"utterance": utterance, "intent": pred.top_intent})

    return results_list

# ── Language Studio Setup Guide ────────────────────────────
def print_clu_guide():
    print("\n" + "="*60)
    print("  CLU PROJECT SETUP — LANGUAGE STUDIO")
    print("="*60)
    guide = [
        "1. language.cognitive.azure.com → Create project",
        "2. Select: Conversational Language Understanding",
        "3. Add INTENTS (e.g., BookFlight, CheckWeather, Cancel)",
        "4. Add ENTITIES (e.g., destination, date, time)",
        "   Entity types: Prebuilt, Learned, List, Regex",
        "5. Add UTTERANCES per intent (min 15 recommended)",
        "6. Label entity spans in each utterance",
        "7. Train → Quick training or Standard training",
        "8. Evaluate: check Precision, Recall, F1 per intent",
        "9. Deploy to named deployment slot",
        "10. Test in Language Studio portal before calling API",
    ]
    for item in guide:
        print(f"  {item}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print_clu_guide()

    # Test utterances — adjust to match your CLU project intents
    test_utterances = [
        "Book a flight to Dar es Salaam on Monday morning",
        "What's the weather like in Nairobi tomorrow?",
        "Cancel my reservation please",
        "I want to check in to my hotel room",
        "Find me the nearest cybersecurity training centre",
    ]

    # Single utterance analysis
    analyze_utterance(test_utterances[0])

    # Multi-turn conversation
    multi_turn_conversation(test_utterances[:3])

    # Batch analysis
    batch_analyze(test_utterances)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • CLU REPLACES LUIS — LUIS retired September 2025")
    print("  • Uses ConversationAnalysisClient (different from TextAnalyticsClient)")
    print("  • kind='Conversation' for CLU projects")
    print("  • prediction.top_intent = highest confidence intent")
    print("  • Entity types: Prebuilt, Learned, List, Regex, Pattern.Any")
    print("  • Orchestration workflow: routes to QA or CLU based on intent")
    print("="*60 + "\n")
