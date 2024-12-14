import os
import json
import numpy as np
import jieba.posseg as pseg
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import mysql.connector
from rank_bm25 import BM25Okapi
import keras as kr
import tensorflow as tf

# 初始化日志记录
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 全局参数
SIMILARITY_THRESHOLD = 1.2  # BM25 相似度阈值
MAX_SEQUENCE_LENGTH = 20    # 模型输入的最大序列长度
BM25_BATCH_SIZE = 500       # 每次加载文档的数量
DB_LIMIT = 5000             # 数据库最大文档数

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "35.185.148.251"),
    "user": os.getenv("DB_USER", "tps"),
    "password": os.getenv("DB_PASSWORD", "0423"),
    "database": os.getenv("DB_NAME", "fake_news_db")
}

# 全局变量
bm25 = None
model = None
tokenizer = None

# 数据库连接
def get_database_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        logging.info("数据库连接成功。")
        return connection
    except mysql.connector.Error as err:
        logging.error(f"数据库连接失败: {err}")
        return None

# 加载分词器
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']  # 返回分词后的列表

# 分块加载文档并初始化 BM25
def initialize_bm25():
    global bm25
    tokenized_corpus = []
    doc_ids = []

    connection = get_database_connection()
    if not connection:
        raise RuntimeError("数据库连接失败，无法初始化 BM25。")

    try:
        cursor = connection.cursor(dictionary=True)
        offset = 0

        while True:
            # 按批次加载文档
            query = f"SELECT id, title, content FROM cleaned_file LIMIT {BM25_BATCH_SIZE} OFFSET {offset}"
            cursor.execute(query)
            batch = cursor.fetchall()
            if not batch:
                break  # 如果没有更多数据，退出循环

            for record in batch:
                text = record["title"] + " " + record["content"]
                tokenized_corpus.append(jieba_tokenizer(text))
                doc_ids.append(record["id"])

            offset += BM25_BATCH_SIZE

        # 初始化 BM25 模型
        bm25 = BM25Okapi(tokenized_corpus)
        bm25.doc_ids = doc_ids  # 保存文档 ID
        logging.info(f"BM25 模型已加载，文档总数: {len(tokenized_corpus)}")
    finally:
        cursor.close()
        connection.close()

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

# 初始化 BM25 和 LSTM
initialize_bm25()
load_model_and_tokenizer()

# 获取最佳匹配项（BM25）
def get_best_match_bm25(input_text):
    global bm25
    if bm25 is None:
        raise RuntimeError("BM25 模型尚未加载。")

    input_tokens = jieba_tokenizer(input_text)
    scores = bm25.get_scores(input_tokens)
    best_index = np.argmax(scores)
    best_score = scores[best_index]

    if best_score >= SIMILARITY_THRESHOLD:
        return bm25.doc_ids[best_index], best_score
    return None, None

# 查询数据库记录
def get_database_record(doc_id):
    connection = get_database_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            query = "SELECT * FROM cleaned_file WHERE id = %s"
            cursor.execute(query, (doc_id,))
            record = cursor.fetchone()
            cursor.close()
            connection.close()
            return record
        except Exception as e:
            logging.error(f"查询数据库记录失败: {e}")
    return None

# 文本预处理
def preprocess_texts(title):
    if tokenizer is None:
        raise ValueError("分词器尚未加载。")
    title_tokenized = jieba_tokenizer(title)
    x_test = tokenizer.texts_to_sequences([" ".join(title_tokenized)])
    x_test = kr.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    return x_test

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
        data = request.json
        logging.info(f"收到的请求数据: {data}")
        input_title = data.get('title', '').strip()

        if not input_title:
            return jsonify({'error': '需要提供标题'}), 400

        if len(input_title) < 3:
            return jsonify({'error': '标题过短'}), 400

        # 使用 BM25 获取最佳匹配
        best_doc_id, best_score = get_best_match_bm25(input_title)
        if not best_doc_id:
            return jsonify({'error': '没有找到足够相似的数据'}), 404

        # 查询数据库记录
        best_match = get_database_record(best_doc_id)
        if not best_match:
            return jsonify({'error': '无法获取匹配的数据库记录'}), 500

        # 使用 LSTM 模型进行预测
        probabilities = predict_category(input_title, best_match["title"])
        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]  # 中文分类
        category = categories[category_index]

        response = {
            'input_title': input_title,
            'matched_title': best_match["title"],
            'matched_content': best_match["content"],
            'bm25_score': best_score,
            'category': category,
            'probabilities': {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)}
        }

        logging.info(f"API 处理总时间: {time.time() - start_time:.4f} 秒")
        return jsonify(response)

    except Exception as e:
        logging.error(f"发生错误: {e}")
        return jsonify({'error': str(e)}), 500

# API 路由：获取历史记录
@app.route('/history', methods=['GET'])
def get_history():
    try:
        connection = get_database_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT id, query_text, result_title, bm25_score, result_category, created_at
                FROM history
                ORDER BY created_at DESC
                LIMIT 20
            """
            cursor.execute(query)
            history_data = cursor.fetchall()
            cursor.close()
            connection.close()
            return jsonify({'history': history_data})
        else:
            return jsonify({'error': '无法连接到数据库'}), 500
    except Exception as e:
        logging.error(f"获取历史记录失败: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
