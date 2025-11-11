import os

from flask import Flask
from flask import request, jsonify
from src.translator import translate_content
from src.llm import configure_client_from_env

app = Flask(__name__)

try:
    configure_client_from_env()
except Exception as e:
    print(f"Warning: Could not configure LLM client: {e}")
    print("The service will not work without a configured LLM client.")

@app.route("/translate/", methods=["POST"])
def translator():
    data = request.get_json()
    content = data.get("content", "")
    is_english, translated_content = translate_content(content)
    return jsonify({
        "is_english": is_english,
        "translated_content": translated_content,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
