import os
import json
import keras as kr
import numpy as np
import tensorflow as tf
import jieba.posseg as pseg
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from rank_bm25 import BM25Okapi  # BM25 实现
import logging
import time  # 引入时间模块用于计时

# 初始化日志记录
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 全局参数
MAX_SEQUENCE_LENGTH = 20
SIMILARITY_THRESHOLD = 1.5  # BM25 阈值，通常调整为 1.2 到 2.0 的范围

# 模型及数据的相对路径
MODEL_PATH = os.getenv("MODEL_PATH", "FNCwithLSTM.h5")
WORD_INDEX_PATH = os.getenv("WORD_INDEX_PATH", "word_index.json")

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "35.185.148.251"),
    "user": os.getenv("DB_USER", "tps"),
    "password": os.getenv("DB_PASSWORD", "0423"),
    "database": os.getenv("DB_NAME", "fake_news_db")
}

# 全局变量
bm25 = None
corpus = None

# 数据库连接池
def get_database_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        logging.info("Database connection established.")
        return connection
    except mysql.connector.Error as err:
        logging.error(f"Database connection error: {err}")
        return None

# 加载已训练的模型
try:
    start_time = time.time()
    model = kr.models.load_model(MODEL_PATH)
    logging.info(f"Model loaded successfully. Time taken: {time.time() - start_time:.4f} seconds")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# 加载 word_index.json 并还原 Tokenizer
try:
    start_time = time.time()
    with open(WORD_INDEX_PATH, 'r', encoding='utf-8') as f:
        word_index = json.load(f)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)  # 添加 num_words 参数
    tokenizer.word_index = word_index
    tokenizer.index_word = {index: word for word, index in word_index.items()}
    logging.info(f"Tokenizer restored successfully. Time taken: {time.time() - start_time:.4f} seconds")
except Exception as e:
    logging.error(f"Error loading tokenizer: {e}")
    tokenizer = None

# 分词函数
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']  # 返回分词后的列表

# 分批加载数据以降低内存使用
def get_corpus_batches(batch_size=500, limit=5000):
    connection = get_database_connection()
    if not connection:
        raise RuntimeError("Failed to connect to the database.")
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(f"SELECT title, content FROM cleaned_file LIMIT {limit}")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [jieba_tokenizer(row["title"] + " " + row["content"]) for row in rows]
    finally:
        cursor.close()
        connection.close()

# 延迟初始化 BM25
def initialize_bm25():
    global bm25
    corpus_batches = get_corpus_batches(batch_size=500, limit=5000)
    tokenized_corpus = []
    for batch in corpus_batches:
        tokenized_corpus.extend(batch)
    bm25 = BM25Okapi(tokenized_corpus)
    logging.info(f"BM25 initialized with {len(tokenized_corpus)} documents.")

# 初始化时调用优化的 BM25 函数
initialize_bm25()

# 预处理函数
def preprocess_texts(title):
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized.")
    title_tokenized = jieba_tokenizer(title)
    x_test = tokenizer.texts_to_sequences([" ".join(title_tokenized)])
    x_test = kr.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    return x_test

# 模型预测
def predict_category(input_title, database_title):
    if model is None:
        raise ValueError("Model is not loaded.")
    input_processed = preprocess_texts(input_title)
    db_processed = preprocess_texts(database_title)
    predictions = model.predict([input_processed, db_processed])
    return predictions

# 使用已初始化的 BM25 进行匹配
def get_best_match_bm25(input_title):
    input_tokens = jieba_tokenizer(input_title)
    start_time = time.time()
    scores = bm25.get_scores(input_tokens)
    logging.info(f"BM25 scoring time: {time.time() - start_time:.4f} seconds")

    best_index = np.argmax(scores)
    best_score = scores[best_index]
    if best_score >= SIMILARITY_THRESHOLD:
        return best_index, best_score
    return None, None

# 增加历史记录的 API
@app.route('/history', methods=['GET'])
def get_history():
    try:
        connection = get_database_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT id, query_text, result_category, result_fake_probability, result_real_probability, created_at
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
            return jsonify({'error': 'Failed to connect to database'}), 500
    except Exception as e:
        logging.error(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500

# API 路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        data = request.json
        logging.info(f"Received request data: {data}")
        input_title = data.get('title', '').strip()

        if not input_title:
            return jsonify({'error': 'Title is required'}), 400

        if len(input_title) < 3:
            return jsonify({'error': 'Title is too short'}), 400

        # 获取最佳匹配项
        best_index, best_score = get_best_match_bm25(input_title)
        if best_index is None:
            return jsonify({'error': 'No sufficiently similar data found in the database'}), 404

        # 使用模型进行预测
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(f"SELECT title FROM cleaned_file LIMIT 5000")
        rows = cursor.fetchall()
        best_match = rows[best_index]
        probabilities = predict_category(input_title, best_match["title"])
        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]  # 中文分類
        category = categories[category_index]

        response = {
            'input_title': input_title,
            'matched_title': best_match["title"],
            'bm25_score': best_score,
            'category': category,
            'probabilities': {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)}
        }

        # 保存历史记录
        query = """
            INSERT INTO history (query_text, result_category, result_fake_probability, result_real_probability)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (
            input_title,
            category,
            response['probabilities']["同意"],
            response['probabilities']["無關"]
        ))
        connection.commit()
        cursor.close()
        connection.close()

        logging.info(f"Total API processing time: {time.time() - start_time:.4f} seconds")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # 从环境变量中获取 Render 分配的端口
    app.run(host='0.0.0.0', port=port)        # 使用动态端口
