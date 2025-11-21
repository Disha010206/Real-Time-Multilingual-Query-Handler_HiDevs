import time
from collections import defaultdict

from flask import Flask, render_template, request, jsonify
from langdetect import detect
from transformers import pipeline

app = Flask(__name__)

translator_nllb = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    max_length=256
)
intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
t5_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)


metrics = {
    "total_queries": 0,
    "avg_latency": 0.0,
    "languages_seen": defaultdict(int),
    "feedback": {"good": 0, "bad": 0},
}


def update_metrics(lang: str, latency_ms: float):
    metrics["total_queries"] += 1
    metrics["languages_seen"][lang] += 1
    prev = metrics["avg_latency"]
    total = metrics["total_queries"]
    metrics["avg_latency"] = ((prev * (total - 1)) + latency_ms) / total



LANGDETECT_TO_NLLB = {
    # Indian languages
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "kn": "kan_Knda",
    "ur": "urd_Arab",

    # Foreign languages
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "ar": "ara_Arab",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh-cn": "zho_Hans",
}


def translate_text(text: str, lang_code: str) -> str:
    """
    Use NLLB with src_lang -> English.
    Fallback to original if needed.
    """

    if lang_code == "en":
        return text

    src_lang = LANGDETECT_TO_NLLB.get(lang_code)

    try:
        if src_lang:
            out = translator_nllb(
                text, src_lang=src_lang, tgt_lang="eng_Latn"
            )[0]["translation_text"]
        else:
            out = translator_nllb(text)[0]["translation_text"]
    except:
        out = text

    if not out.strip() or out.strip() == text.strip():
        return text

    return out



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    start_time = time.time()


    try:
        lang_code = detect(query)
    except:
        lang_code = "unknown"

    lang_display_map = {
        "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
        "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Punjabi",
        "kn": "Kannada", "ur": "Urdu",
        "en": "English", "fr": "French", "es": "Spanish", "de": "German",
        "ar": "Arabic", "ru": "Russian", "zh-cn": "Chinese", "ja": "Japanese",
        "ko": "Korean",
    }
    lang_display = lang_display_map.get(lang_code, lang_code)

    
    translation = translate_text(query, lang_code)

    resp_prompt = (
        "You are a customer support agent. "
        "Write a helpful, polite reply (2–3 sentences). "
        "Do NOT repeat the customer's message. "
        "Give empathy + next steps.\n\n"
        f"Customer message: \"{translation}\""
    )

    response_output = t5_model(
        resp_prompt,
        max_length=120,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )[0]["generated_text"]

    
    latency_ms = round((time.time() - start_time) * 1000, 1)
    update_metrics(lang_display, latency_ms)

    return jsonify({
        "detected_language": lang_display,
        "translation": translation,
        "suggested_response": response_output,
        "latency_ms": latency_ms,
        "metrics": {
            "total_queries": metrics["total_queries"],
            "avg_latency_ms": round(metrics["avg_latency"], 1),
            "languages": dict(metrics["languages_seen"]),
            "feedback": metrics["feedback"],
        },
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    fb = (request.get_json() or {}).get("feedback")
    if fb == "good":
        metrics["feedback"]["good"] += 1
    elif fb == "bad":
        metrics["feedback"]["bad"] += 1
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
