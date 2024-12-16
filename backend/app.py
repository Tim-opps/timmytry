import os
import json
import numpy as np
import pandas as pd
import jieba
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import keras as kr
import tensorflow as tf
from flask import send_from_directory，render_template

# 初始化日志记录
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 全局参数
MAX_SEQUENCE_LENGTH = 20    # 模型输入的最大序列长度
SIMILARITY_THRESHOLD = 0.5  # 数据匹配的最低分数
CSV_FILE = "datacombined_1_tokenized.csv"  # 分词后的 CSV 文件路径

# 全局变量
model = None
tokenizer = None
data = None  # 存储分词后的 CSV 数据

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
        logging.info("\u5206\u8bcd\u5668\u52a0\u8f7d\u6210\u529f。")
    except Exception as e:
        logging.error(f"\u52a0\u8f7d LSTM \u6a21\u578b\u6216\u5206\u8bcd\u5668\u5931\u8d25: {e}")

load_model_and_tokenizer()

# 加载 CSV 数据
def load_csv_data():
    global data
    try:
        data = pd.read_csv(CSV_FILE, dtype={"tokenized_title": "string", "tokenized_content": "string"})
        logging.info(f"\u5206\u8bcd\u540e\u7684 CSV \u6587\u4ef6\u5df2\u52a0\u8f7d\uff0c\u5171 {len(data)} \u6761\u8bb0\u5f55。")
    except Exception as e:
        logging.error(f"\u52a0\u8f7d CSV \u6587\u4ef6\u5931\u8d25: {e}")

load_csv_data()

# CSV 匹配函数
def find_best_match(input_text):
    best_match = None
    best_score = 0

    # 对用户输入进行 jieba 分词，并转为空格分隔的字组
    input_tokens = set(" ".join(jieba.lcut(input_text)).split())

    for _, row in data.iterrows():
        title_tokens = set(row['tokenized_title'].split())
        content_tokens = set(row['tokenized_content'].split())
        combined_text = title_tokens | content_tokens  # 合并标题和内容分词

        # 匹配分数计算
        common_tokens = input_tokens & combined_text
        score = len(common_tokens) / len(input_tokens) if input_tokens else 0

        if score > best_score and score >= SIMILARITY_THRESHOLD:
            best_score = score
            best_match = row

    return best_match, best_score

# 文本预处理
def preprocess_texts(text):
    if tokenizer is None:
        raise ValueError("\u5206\u8bcd\u5668\u672a\u52a0\u8f7d\u3002")
    # 对文本进行 jieba 分词
    tokenized_text = " ".join(jieba.lcut(text))
    x_test = tokenizer.texts_to_sequences([tokenized_text])
    x_test = kr.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    return x_test

# 模型预测
def predict_category(input_title, database_title):
    if model is None:
        raise ValueError("LSTM \u6a21\u578b\u672a\u52a0\u8f7d\u3002")
    input_processed = preprocess_texts(input_title)
    db_processed = preprocess_texts(database_title)
    predictions = model.predict([input_processed, db_processed])
    return predictions

# API 路由：预测
@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        logging.info("\u5f00\u59cb\u5904\u7406 /predict \u8bf7\u6c42")

        # 1. 解析请求数据
        data_request = request.json
        input_title = data_request.get('title', '').strip()

        if not input_title or len(input_title) < 3:
            logging.warning("\u65e0\u6548\u8f93\u5165\u6807\u9898")
            return jsonify({'error': '请提供有效的标题'}), 400

        logging.info(f"\u63a5\u6536\u5230\u7684\u8f93\u5165\u6807\u9898: {input_title}")

        # 2. 匹配 CSV 数据
        best_match, best_score = find_best_match(input_title)
        if best_match is None:
            logging.info("\u672a\u627e\u5230\u5339\u914d\u6570\u636e")
            return jsonify({'error': '未找到匹配数据'}), 404

        # 3. 模型预测
        logging.info("\u5f00\u59cb\u8fdb\u884c\u6a21\u578b\u9884\u6d4b")
        probabilities = predict_category(input_title, best_match["tokenized_title"])
        category_index = np.argmax(probabilities)
        categories = ["无关", "同意", "不同意"]
        category = categories[category_index]

        # 4. 构建响应数据
        response = {
            'input_title': input_title,
            'matched_title': best_match["tokenized_title"],
            'matched_content': best_match["tokenized_content"],
            'match_score': round(best_score, 4),
            'category': category,
            'classification': "假消息" if int(best_match["classification"]) == 1 else "真消息",
            'probabilities': {cat: round(float(probabilities[0][i]), 4) for i, cat in enumerate(categories)}
        }

        logging.info(f"API \u5904\u7406\u603b\u65f6\u95f4: {time.time() - start_time:.4f} 秒")
        return jsonify(response)

    except Exception as e:
        logging.error(f"\u53d1\u751f\u9519\u8bef: {e}")
        return jsonify({'error': '服务器内部错误', 'message': str(e)}), 500


@app.route('/static/<path:path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', datacombined_1_processed.csv)


# 渲染首頁（或其他頁面）
@app.route('/random')
def random_page():
    return render_template('random.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)






