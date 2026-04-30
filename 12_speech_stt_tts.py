"""
============================================================
AI-102 | Program 12 — Speech-to-Text & Text-to-Speech
Service : Azure AI Speech
Skill   : Process and translate speech
============================================================
Features:
  • Speech-to-Text (STT): microphone & file input
  • Text-to-Speech (TTS): neural voices + SSML
  • Continuous recognition
  • Speech translation (speech → translated text)
  • Speaker recognition concepts
============================================================
"""

import os
import time
import azure.cognitiveservices.speech as speechsdk

SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY", "<your-speech-key>")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")

def get_speech_config(language: str = "en-US") -> speechsdk.SpeechConfig:
    config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    config.speech_recognition_language = language
    return config

# ── 1. Speech-to-Text from Microphone ─────────────────────
def stt_from_microphone(language: str = "en-US") -> str:
    """
    Capture single utterance from default microphone.
    Stops automatically after silence is detected.
    """
    speech_config = get_speech_config(language)
    audio_config  = speechsdk.AudioConfig(use_default_microphone=True)
    recognizer    = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    print("\n" + "="*60)
    print("  SPEECH-TO-TEXT — MICROPHONE")
    print("="*60)
    print(f"  Language: {language}")
    print("  Listening... speak now (stops on silence)")

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"  Recognised: {result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print(f"  No speech detected: {result.no_match_details}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.CancellationDetails(result)
        print(f"  Cancelled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"  Error: {cancellation.error_details}")
    return ""

# ── 2. Speech-to-Text from Audio File ─────────────────────
def stt_from_file(audio_file_path: str, language: str = "en-US") -> str:
    """
    Transcribe speech from a WAV audio file.
    Supports: WAV, MP3, OGG, FLAC formats.
    """
    speech_config = get_speech_config(language)
    audio_config  = speechsdk.AudioConfig(filename=audio_file_path)
    recognizer    = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    print("\n" + "="*60)
    print("  SPEECH-TO-TEXT — FROM FILE")
    print("="*60)
    print(f"  File    : {audio_file_path}")
    print(f"  Language: {language}")

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"  Recognised : {result.text}")
        print(f"  Duration   : {result.duration}")
        print(f"  Offset     : {result.offset}")
        return result.text
    else:
        print(f"  Result: {result.reason}")
        return ""

# ── 3. Continuous Speech Recognition ─────────────────────
def continuous_recognition(duration_seconds: int = 10, language: str = "en-US") -> list[str]:
    """
    Continuously recognise speech over a longer period.
    Good for longer audio files or live streaming.
    """
    speech_config = get_speech_config(language)
    audio_config  = speechsdk.AudioConfig(use_default_microphone=True)
    recognizer    = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    recognised_texts = []
    done = False

    def recognised_handler(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognised_texts.append(evt.result.text)
            print(f"  ✓ {evt.result.text}")

    def cancelled_handler(evt):
        nonlocal done
        print(f"  Cancelled: {evt.reason}")
        done = True

    def session_stopped_handler(evt):
        nonlocal done
        done = True

    # Wire up event handlers
    recognizer.recognized.connect(recognised_handler)
    recognizer.canceled.connect(cancelled_handler)
    recognizer.session_stopped.connect(session_stopped_handler)

    print("\n" + "="*60)
    print("  CONTINUOUS SPEECH RECOGNITION")
    print("="*60)
    print(f"  Listening for {duration_seconds} seconds...")

    recognizer.start_continuous_recognition()
    time.sleep(duration_seconds)
    recognizer.stop_continuous_recognition()

    print(f"\n  Full transcript:")
    full_text = " ".join(recognised_texts)
    print(f"  {full_text}")
    return recognised_texts

# ── 4. Text-to-Speech — Basic ─────────────────────────────
def tts_basic(text: str, output_file: str = "output_basic.wav",
              voice: str = "en-US-JennyNeural") -> None:
    """
    Convert text to speech using a Neural voice.
    Saves output to WAV file.
    """
    speech_config = get_speech_config()
    speech_config.speech_synthesis_voice_name = voice

    audio_config  = speechsdk.AudioConfig(filename=output_file)
    synthesizer   = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    print("\n" + "="*60)
    print("  TEXT-TO-SPEECH — BASIC")
    print("="*60)
    print(f"  Voice  : {voice}")
    print(f"  Text   : {text[:80]}")

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"  ✅ Audio saved: {output_file}")
        print(f"  Duration : {result.audio_duration}")
        print(f"  Bytes    : {len(result.audio_data)}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.SpeechSynthesisCancellationDetails(result)
        print(f"  ❌ Cancelled: {details.reason}")
        print(f"  Error: {details.error_details}")

# ── 5. Text-to-Speech with SSML ───────────────────────────
def tts_with_ssml(output_file: str = "output_ssml.wav") -> None:
    """
    Fine-grained control over speech using SSML markup.
    SSML controls: voice, rate, pitch, volume, pauses, emphasis, language.
    """
    speech_config = get_speech_config()
    audio_config  = speechsdk.AudioConfig(filename=output_file)
    synthesizer   = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    ssml = """
    <speak version='1.0'
           xmlns='http://www.w3.org/2001/10/synthesis'
           xmlns:mstts='https://www.w3.org/2001/mstts'
           xml:lang='en-US'>

        <voice name='en-US-JennyNeural'>
            <mstts:express-as style='customerservice'>
                Welcome to the Azure AI Foundry training.
                <break time='500ms'/>
                Today we will cover the following topics.
            </mstts:express-as>
        </voice>

        <voice name='en-US-GuyNeural'>
            <prosody rate='medium' pitch='+5%' volume='loud'>
                First: Natural Language Processing.
                <emphasis level='strong'>This is critical for AI-102.</emphasis>
            </prosody>
        </voice>

        <voice name='en-US-JennyNeural'>
            <prosody rate='slow' pitch='-10%'>
                Take your time. Understand each concept thoroughly.
            </prosody>
            <break time='1s'/>
            <say-as interpret-as='ordinal'>1</say-as> through
            <say-as interpret-as='ordinal'>10</say-as> programs
            are all you need to pass the exam.
        </voice>
    </speak>
    """

    print("\n" + "="*60)
    print("  TEXT-TO-SPEECH WITH SSML")
    print("="*60)
    print("  SSML features used: express-as, break, prosody, emphasis, say-as")

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"  ✅ SSML audio saved: {output_file}")
    else:
        details = speechsdk.SpeechSynthesisCancellationDetails(result)
        print(f"  ❌ Error: {details.error_details}")

# ── 6. List Available Voices ──────────────────────────────
def list_voices(locale: str = "en-US") -> None:
    """
    List all available neural voices for a given locale.
    """
    speech_config = get_speech_config()
    synthesizer   = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    result = synthesizer.get_voices_async(locale).get()

    print("\n" + "="*60)
    print(f"  AVAILABLE VOICES — {locale}")
    print("="*60)

    if result.reason == speechsdk.ResultReason.VoicesListRetrieved:
        for voice in result.voices:
            print(f"  {voice.short_name:<35} "
                  f"[{voice.gender.name:<8}] "
                  f"Style: {', '.join(voice.style_list) if voice.style_list else 'default'}")
    else:
        print(f"  Error: {result.error_details}")

# ── 7. Speech Translation ─────────────────────────────────
def translate_speech(from_language: str = "en-US",
                     to_languages: list[str] = ["fr", "hi", "sw"],
                     audio_file: str = None) -> None:
    """
    Translate speech from one language to text in another.
    Speech → recognised → translated text output.
    """
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    translation_config.speech_recognition_language = from_language

    for lang in to_languages:
        translation_config.add_target_language(lang)

    if audio_file:
        audio_config = speechsdk.AudioConfig(filename=audio_file)
    else:
        audio_config = speechsdk.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config,
        audio_config=audio_config
    )

    print("\n" + "="*60)
    print("  SPEECH TRANSLATION")
    print("="*60)
    print(f"  From: {from_language} → To: {to_languages}")
    print("  Listening...")

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        print(f"\n  Recognised ({from_language}): {result.text}")
        print(f"  Translations:")
        for lang, text in result.translations.items():
            print(f"    [{lang}]: {text}")
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = speechsdk.translation.CancellationDetails(result)
        print(f"  Cancelled: {details.reason}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # TTS demos (don't require microphone)
    tts_basic(
        text="Welcome to the Azure AI Speech service demonstration. "
             "This program covers text to speech conversion for the AI-102 exam.",
        output_file="output_basic.wav",
        voice="en-US-JennyNeural"
    )

    tts_with_ssml(output_file="output_ssml.wav")

    list_voices(locale="en-US")

    # STT demos — uncomment when microphone is available
    # stt_from_microphone(language="en-US")
    # continuous_recognition(duration_seconds=10)

    # Speech translation — uncomment when mic is available
    # translate_speech(from_language="en-US", to_languages=["fr", "hi", "sw"])

    print("\n  KEY POINTS FOR AI-102:")
    print("  • SpeechConfig needs subscription KEY + REGION (not endpoint)")
    print("  • recognize_once_async() → single utterance")
    print("  • start_continuous_recognition() → long-form audio")
    print("  • speak_text_async() → basic TTS")
    print("  • speak_ssml_async() → advanced TTS with full SSML control")
    print("  • SSML tags: <prosody>, <break>, <emphasis>, <say-as>, <voice>")
    print("  • express-as styles: newscast, customerservice, cheerful, sad")
    print("  • SpeechTranslationConfig for speech → translated text")
    print("  • Neural voices identified as: 'en-US-JennyNeural' format")
    print("="*60 + "\n")
