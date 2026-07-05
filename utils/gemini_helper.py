import os
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except Exception:
    genai = None

API_KEY = os.getenv("GEMINI_API_KEY")


def _configure_genai():
    if genai is None:
        return None
    if not API_KEY:
        return None
    try:
        genai.configure(api_key=API_KEY)
        # safe default model name; may need update per your account
        return genai.GenerativeModel("models/gemini-flash-latest")
    except Exception:
        return None


def _build_prompt(primary, secondary, confidence, text, field=None):
    """Construct a detailed, expert, empathetic prompt for Gemini."""
    prompt = [
        "You are an empathetic learning coach.",
        f"Primary emotion: {primary}.",
    ]
    if secondary:
        prompt.append(f"Secondary emotion: {secondary} (confidence {confidence:.2f}).")
    if field:
        prompt.append(f"Student field: {field}.")
    prompt.append("Student message:")
    prompt.append(text)
    prompt.append("\nProvide a short, compassionate, actionable study-focused response: 3 bullet-point strategies and a gentle closing sentence.")
    return "\n".join(prompt)


def get_support_response(primary_emotion, secondary_emotion, confidence, text, field=None, regenerate=False):
    """Generate supportive learning advice using Gemini; fallback to a local, user-safe template if the API is unavailable."""
    model = _configure_genai()
    prompt = _build_prompt(primary_emotion, secondary_emotion, confidence, text, field)

    def _fallback_template() -> str:
        lines = [
            f"I hear that you're feeling {primary_emotion}.",
            "Here are three gentle, practical steps you can try:",
            "1. Break the task into very small chunks and set a 10-minute timer.",
            "2. Find a low-pressure way to connect with classmates (ask a single question).",
            "3. Switch to an audio or visual learning resource for 20 minutes.",
            "If you'd like, I can generate a study plan or rephrase this advice for your specific subject.",
        ]
        return "\n".join(lines)

    if model is None:
        return _fallback_template()

    try:
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            return str(response.candidates[0])
        return _fallback_template()
    except Exception:
        return _fallback_template()