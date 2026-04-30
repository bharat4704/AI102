"""
============================================================
AI-102 | Program 10 — Question Answering (Custom QA)
Service : Azure AI Language
Skill   : Build custom NLP solutions
============================================================
Custom QA replaces QnA Maker (retired March 2025).
Builds knowledge bases from: URLs, files, FAQs, chitchat.
Supports: Precise answers, follow-up prompts, multi-turn QA.
============================================================
"""

import os
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.ai.language.questionanswering.models import (
    QueryParameters,
    KnowledgeBaseAnswer,
)
from azure.core.credentials import AzureKeyCredential

ENDPOINT       = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://<your-resource>.cognitiveservices.azure.com/")
KEY            = os.getenv("AZURE_LANGUAGE_KEY", "<your-key>")
QA_PROJECT     = os.getenv("QA_PROJECT_NAME", "<your-qa-project>")
QA_DEPLOYMENT  = os.getenv("QA_DEPLOYMENT_NAME", "production")

def get_client():
    return QuestionAnsweringClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY)
    )

# ── 1. Basic Question Answering ───────────────────────────
def ask_question(question: str) -> None:
    """
    Query the knowledge base with a question.
    Returns top answers with confidence scores and source.
    """
    client = get_client()

    result = client.get_answers(
        question=question,
        project_name=QA_PROJECT,
        deployment_name=QA_DEPLOYMENT,
        top=3,                              # Return top 3 answers
        confidence_threshold=0.3,           # Minimum confidence (0.0 - 1.0)
    )

    print("\n" + "="*65)
    print("  QUESTION ANSWERING")
    print("="*65)
    print(f"\n  Question: '{question}'")
    print(f"  Answers  ({len(result.answers)} returned):\n")

    for i, answer in enumerate(result.answers):
        print(f"  Answer {i+1}:")
        print(f"    Text       : {answer.answer}")
        print(f"    Confidence : {answer.confidence:.4f}")
        print(f"    Source     : {answer.source}")
        print(f"    QA ID      : {answer.qna_id}")
        if answer.short_answer:
            print(f"    Short Ans  : {answer.short_answer.text} "
                  f"(conf: {answer.short_answer.confidence:.2f})")
        print()

# ── 2. Precise Answer Extraction ─────────────────────────
def ask_with_precise_answer(question: str) -> None:
    """
    Enable precise answer extraction — returns a short span
    from within the answer text (like highlighted search result).
    """
    client = get_client()

    result = client.get_answers(
        question=question,
        project_name=QA_PROJECT,
        deployment_name=QA_DEPLOYMENT,
        short_answer_options={"enable": True, "size": 1},
        top=1,
        confidence_threshold=0.5,
    )

    print("\n" + "="*65)
    print("  PRECISE ANSWER EXTRACTION")
    print("="*65)
    print(f"\n  Question: '{question}'")

    for answer in result.answers:
        print(f"  Full Answer  : {answer.answer}")
        if answer.short_answer:
            print(f"  Precise Span : '{answer.short_answer.text}'")
            print(f"  Confidence   : {answer.short_answer.confidence:.4f}")

# ── 3. Multi-Turn QA (Follow-up Prompts) ─────────────────
def multi_turn_qa(initial_question: str) -> None:
    """
    Simulate multi-turn conversation with follow-up prompts.
    The answer may contain prompts for clarification/follow-up.
    """
    client = get_client()

    print("\n" + "="*65)
    print("  MULTI-TURN QA — FOLLOW-UP PROMPTS")
    print("="*65)
    print(f"\n  Initial Question: '{initial_question}'")

    result = client.get_answers(
        question=initial_question,
        project_name=QA_PROJECT,
        deployment_name=QA_DEPLOYMENT,
        top=1,
        confidence_threshold=0.3,
    )

    if result.answers:
        answer = result.answers[0]
        print(f"  Answer: {answer.answer[:200]}")

        if answer.dialog and answer.dialog.prompts:
            print(f"\n  Follow-up prompts available:")
            for prompt in answer.dialog.prompts:
                print(f"    [{prompt.display_order}] {prompt.display_text}")
                print(f"         → QA ID: {prompt.qna_id}")

            # Simulate user selecting first follow-up
            if answer.dialog.prompts:
                follow_up_id = answer.dialog.prompts[0].qna_id
                print(f"\n  User selects: '{answer.dialog.prompts[0].display_text}'")

                follow_result = client.get_answers_from_text(
                    question=answer.dialog.prompts[0].display_text,
                    text_documents=[],
                )

# ── 4. Answer from Text (No KB needed) ───────────────────
def answer_from_text(question: str, text_documents: list[str]) -> None:
    """
    Answer questions directly from provided text — no knowledge base needed.
    Useful for document Q&A on the fly.
    """
    from azure.ai.language.questionanswering.models import TextDocument

    client = get_client()

    text_docs = [
        TextDocument(id=str(i), text=doc)
        for i, doc in enumerate(text_documents)
    ]

    result = client.get_answers_from_text(
        question=question,
        text_documents=text_docs,
        language="en"
    )

    print("\n" + "="*65)
    print("  ANSWER FROM TEXT (No Knowledge Base)")
    print("="*65)
    print(f"\n  Question: '{question}'")
    print(f"  Source text provided: {len(text_documents)} document(s)\n")

    for answer in result.answers:
        if answer.confidence > 0.3:
            print(f"  Answer     : {answer.answer}")
            print(f"  Confidence : {answer.confidence:.4f}")
            if answer.short_answer:
                print(f"  Precise    : '{answer.short_answer.text}'")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is Azure AI Foundry?",
        "How do I enable multi-factor authentication?",
        "What is the difference between CLU and LUIS?",
        "How much does Azure Language service cost?",
    ]

    # Inline text for no-KB demo
    context_docs = [
        """Azure AI Foundry is Microsoft's unified AI development platform.
           It provides access to over 1600 foundation models and includes
           tools for fine-tuning, evaluation, and responsible AI.""",
        """Multi-factor authentication (MFA) adds an extra layer of security.
           To enable MFA in Azure: go to Azure Active Directory, select Users,
           then Multi-Factor Authentication, and follow the setup wizard.""",
    ]

    ask_question(questions[0])
    ask_with_precise_answer(questions[1])
    multi_turn_qa(questions[0])
    answer_from_text(questions[0], context_docs)

    print("\n  KEY POINTS FOR AI-102:")
    print("  • Custom QA REPLACES QnA Maker (retired March 2025)")
    print("  • QuestionAnsweringClient (separate from TextAnalyticsClient)")
    print("  • get_answers() → queries knowledge base")
    print("  • get_answers_from_text() → no KB needed, uses provided text")
    print("  • short_answer_options enables precise span extraction")
    print("  • confidence_threshold filters low-quality answers")
    print("  • answer.dialog.prompts → follow-up prompt chaining")
    print("  • KB sources: URLs, .pdf, .docx, .tsv, chitchat datasets")
    print("="*65 + "\n")
