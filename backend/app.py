import os
import json
import numpy as np
import pandas as pd
import jieba.posseg as pseg
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import keras as kr
import tensorflow as tf

# 初始化日志记录
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 全局参数
MAX_SEQUENCE_LENGTH = 20
SIMILARITY_THRESHOLD = 0.5
CSV_FILE = "datacombined_1_tokenized.csv"

# 全局变量
model = None
tokenizer = None
data = None

# 加载 LSTM 模型和分词器
def load_model_and_tokenizer():
    global model, tokenizer
    try:
        start_time = time.time()
        model_path = os.getenv("MODEL_PATH", "FNCwithLSTM.h5")
        word_index_path = os.getenv("WORD_INDEX_PATH", "word_index.json")

        model = kr.models.load_model(model_path)
        logging.info(f"LSTM 模型加载成功，耗时: {time.time() - start_time:.4f} 秒")

        with open(word_index_path, 'r', encoding='utf-8') as f:
            word_index = json.load(f)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
        tokenizer.word_index = word_index
        logging.info("分词器加载成功。")
    except Exception as e:
        logging.error(f"加载 LSTM 模型或分词器失败: {e}")

load_model_and_tokenizer()

# 加载 CSV 数据
def load_csv_data():
    global data
    try:
        data = pd.read_csv(CSV_FILE, dtype={"title": "string", "content": "string"})
        data['combined_text'] = data.apply(lambda row: set(row['title'].split() + row['content'].split()), axis=1)
        logging.info(f"分词后的 CSV 文件已加载，共 {len(data)} 条记录。")
    except Exception as e:
        logging.error(f"加载 CSV 文件失败: {e}")

load_csv_data()

# 分词函数
def jieba_tokenizer(text):
    if not isinstance(text, str) or not text.strip():
        return set()
    words = pseg.cut(text)
    return set([word for word, flag in words if flag != 'x'])

# 匹配函数
def find_best_match(input_text):
    input_tokens = jieba_tokenizer(input_text)
    best_match = None
    best_score = 0

    for row in data.itertuples(index=False):
        combined_text = row.combined_text
        common_tokens = input_tokens & combined_text
        score = len(common_tokens) / len(input_tokens) if len(input_tokens) > 0 else 0

        if score > best_score and score >= SIMILARITY_THRESHOLD:
            best_score = score
            best_match = row

    return best_match, best_score

# 文本预处理
def preprocess_texts(title):
    title_tokenized = jieba_tokenizer(title)
    x_test = tokenizer.texts_to_sequences([" ".join(title_tokenized)])
    return kr.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

# 模型预测
def predict_category(input_title, database_title):
    input_processed = preprocess_texts(input_title)
    db_processed = preprocess_texts(database_title)
    return model.predict([input_processed, db_processed])

# API 路由：预测
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_title = data.get('title', '').strip()
        if not input_title:
            return jsonify({'error': '需要提供标题'}), 400

        best_match, best_score = find_best_match(input_title)
        if best_match is None:
            return jsonify({'error': '没有找到足够相似的数据'}), 404

        probabilities = predict_category(input_title, best_match.title)
        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]
        response = {
            'input_title': input_title,
            'matched_title': best_match.title,
            'category': categories[category_index],
            'probabilities': {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)}
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"发生错误: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
