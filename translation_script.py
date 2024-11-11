import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from google.cloud import storage
import base64
from flask import Flask, request, jsonify

device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

lang_code_map = {
    "fr": "fra_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
}

client = storage.Client(project="core-dev-435517")

def download_blob(bucket_name, source_blob_name, destination_file_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def parse_txt_to_json(input_text):
    json_data = []
    lines = input_text.splitlines()
    for line in lines:
        time, text = line.split("] ", 1)
        time = time.strip("[]")
        json_data.append({"time": time, "text": text})
    return json_data

def translate_nllb(input_data):
    detected_lang = detect(input_data[0]['text'])
    source_lang = lang_code_map.get(detected_lang)

    if detected_lang == 'fr':
        target_langs = ['eng_Latn', 'spa_Latn']
    elif detected_lang == 'es':
        target_langs = ['fra_Latn', 'eng_Latn']
    elif detected_lang == 'en':
        target_langs = ['fra_Latn', 'spa_Latn']

    tokenizer.src_lang = source_lang
    texts = [entry['text'] for entry in input_data]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    translations = {}
    for target_lang in target_langs:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
        translated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations[target_lang] = translated_texts

    translated_data = {}
    for target_lang, translated_texts in translations.items():
        lang_key = list(lang_code_map.keys())[list(lang_code_map.values()).index(target_lang)]
        translated_data[f"translation_{lang_key}"] = [
            {"time": entry["time"], "text": translated_texts[i]}
            for i, entry in enumerate(input_data)
        ]

    return translated_data

@app.route("/translate", methods=["POST"])
def translate_handler():
    try:
        data = request.get_json()

        if "message" in data and "data" in data["message"]:
            event_data = json.loads(base64.b64decode(data['message']['data']).decode('utf-8'))
            input_bucket = event_data['bucket']
            input_blob_name = event_data['name']
        else:
            return jsonify({"error": "Invalid event format"}), 400

        output_bucket = 'vertex-translation-output'

        input_file = '/tmp/translation_input.txt'
        download_blob(input_bucket, input_blob_name, input_file)

        with open(input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()

        input_data = parse_txt_to_json(input_text)
        translated_data = translate_nllb(input_data)

        for lang_key, output_data in translated_data.items():
            output_file = f'/tmp/{lang_key}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            upload_blob(output_bucket, output_file, f"{lang_key}_{input_blob_name}.json")

        return jsonify({"status": "success", "message": f"Translations completed and uploaded for {input_blob_name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)