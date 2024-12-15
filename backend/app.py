import os
import json
import numpy as np
import jieba
import jieba.posseg as pseg
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import pandas as pd
import keras as kr
import tensorflow as tf

# 初始化日志记录
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 全局参数
MAX_SEQUENCE_LENGTH = 20    # 模型输入的最大序列长度
SIMILARITY_THRESHOLD = 0.5  # 数据库匹配的最低分数
CSV_FILE_PATH = "datacombined_1_tokenized.csv"  # 分词后的 CSV 文件

# 全局变量
model = None
tokenizer = None
csv_data = None

# 加载 CSV 数据
try:
    logging.info(f"正在加载 CSV 文件 {CSV_FILE_PATH} ...")
    csv_data = pd.read_csv(CSV_FILE_PATH)
    logging.info(f"CSV 文件加载成功，包含 {len(csv_data)} 条记录。")
except Exception as e:
    logging.error(f"加载 CSV 文件失败：{e}")
    exit()

# 加载 LSTM 模型和分词器
def load_model_and_tokenizer():
    global model, tokenizer
    try:
        start_time = time.time()
        model_path = os.getenv("MODEL_PATH", "FNCwithLSTM.h5")
        word_index_path = os.getenv("WORD_INDEX_PATH", "word_index.json")

        # 加载模型
        model = kr.models.load_model(model_path)
        logging.info(f"LSTM 模型加载成功，耗时: {time.time() - start_time:.4f} 秒")

        # 加载分词器
        with open(word_index_path, 'r', encoding='utf-8') as f:
            word_index = json.load(f)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
        tokenizer.word_index = word_index
        tokenizer.index_word = {index: word for word, index in word_index.items()}
        logging.info("分词器加载成功。")
    except Exception as e:
        logging.error(f"加载 LSTM 模型或分词器失败: {e}")

load_model_and_tokenizer()

# 文本预处理
def preprocess_texts(title):
    if tokenizer is None:
        raise ValueError("分词器尚未加载。")
    title_tokenized = " ".join(jieba.cut(title))
    x_test = tokenizer.texts_to_sequences([title_tokenized])
    x_test = kr.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    return x_test

# CSV 匹配函数
def find_best_match(input_text):
    best_match = None
    best_score = 0

    try:
        input_tokens = set(jieba.cut(input_text))

        for _, row in csv_data.iterrows():
            title_tokens = set(row['tokenized_title'].split())
            content_tokens = set(row['tokenized_content'].split())
            combined_tokens = title_tokens | content_tokens

            # 计算简单的相似度分数
            common_tokens = input_tokens & combined_tokens
            score = len(common_tokens) / len(input_tokens)

            if score > best_score and score >= SIMILARITY_THRESHOLD:
                best_score = score
                best_match = row

        return best_match, best_score

    except Exception as e:
        logging.error(f"匹配过程中发生错误: {e}")
        return None, 0

# 模型预测
def predict_category(input_title, database_title):
    if model is None:
        raise ValueError("LSTM 模型尚未加载。")
    input_processed = preprocess_texts(input_title)
    db_processed = preprocess_texts(database_title)
    predictions = model.predict([input_processed, db_processed])
    return predictions

# API 路由：预测
@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        logging.info("开始处理 /predict 请求")

        # Step 1: 解析请求数据
        data = request.json
        logging.info(f"收到的请求数据: {data}")
        input_title = data.get('title', '').strip()

        if not input_title:
            return jsonify({'error': '需要提供标题'}), 400

        if len(input_title) < 3:
            return jsonify({'error': '标题过短'}), 400

        # Step 2: CSV 匹配
        match_start_time = time.time()
        best_match, best_score = find_best_match(input_title)
        if best_match is None:
            return jsonify({'error': '没有找到足够相似的数据'}), 404

        match_end_time = time.time()
        logging.info(f"CSV 匹配耗时: {match_end_time - match_start_time:.4f} 秒")

        # Step 3: 使用 LSTM 模型进行预测
        model_start_time = time.time()
        probabilities = predict_category(input_title, best_match["tokenized_title"])
        model_end_time = time.time()
        logging.info(f"LSTM 模型预测耗时: {model_end_time - model_start_time:.4f} 秒")

        # Step 4: 准备响应数据
        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]  # 中文分类
        category = categories[category_index]

        response = {
            'input_title': input_title,
            'matched_title': best_match["tokenized_title"],
            'matched_content': best_match["tokenized_content"],
            'match_score': best_score,
            'category': category,
            'probabilities': {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)}
        }

        total_time = time.time() - start_time
        logging.info(f"API 处理总时间: {total_time:.4f} 秒")
        return jsonify(response)

    except Exception as e:
        logging.error(f"发生错误: {e}")
        return jsonify({'error': str(e)}), 500

# API 路由：获取历史记录（示例：仅保留接口结构，不存储历史记录）
@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'message': '当前未启用历史记录存储功能。'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
