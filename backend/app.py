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

# 初始化日志記錄
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 應用
app = Flask(__name__)
CORS(app)

# 全局參數
SIMILARITY_THRESHOLD = 1.2  # BM25 相似度閾值
PRECOMPUTED_BM25_FILE = "bm25_data.json"  # 本地存儲的 BM25 數據

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "35.185.148.251"),
    "user": os.getenv("DB_USER", "tps"),
    "password": os.getenv("DB_PASSWORD", "0423"),
    "database": os.getenv("DB_NAME", "fake_news_db")
}

# 全局變量
bm25 = None
corpus = None
doc_ids = None

# 数据库连接
def get_database_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        logging.info("数据库连接成功。")
        return connection
    except mysql.connector.Error as err:
        logging.error(f"数据库连接失败: {err}")
        return None

# 分词器
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']

# 加载预计算的 BM25 数据
def load_precomputed_bm25():
    global bm25, corpus, doc_ids
    try:
        with open(PRECOMPUTED_BM25_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        corpus = data["corpus"]
        doc_ids = data["doc_ids"]
        bm25 = BM25Okapi(corpus)
        logging.info(f"BM25 模型已加载，文档数量: {len(corpus)}")
    except Exception as e:
        logging.error(f"无法加载 BM25 数据: {e}")

# 初始化 BM25
load_precomputed_bm25()

# 获取最佳匹配项（BM25）
def get_best_match_bm25(input_text):
    global bm25, corpus, doc_ids
    if bm25 is None:
        raise RuntimeError("BM25 模型尚未加载。")

    input_tokens = jieba_tokenizer(input_text)
    scores = bm25.get_scores(input_tokens)
    best_index = np.argmax(scores)
    best_score = scores[best_index]

    if best_score >= SIMILARITY_THRESHOLD:
        return doc_ids[best_index], best_score
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

# API 路由：预测
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_title = data.get('title', '').strip()

        if not input_title:
            return jsonify({'error': '需要提供标题'}), 400

        best_doc_id, best_score = get_best_match_bm25(input_title)
        if not best_doc_id:
            return jsonify({'error': '没有找到足够相似的数据'}), 404

        best_match = get_database_record(best_doc_id)
        if not best_match:
            return jsonify({'error': '无法获取匹配的数据库记录'}), 500

        response = {
            'input_title': input_title,
            'matched_title': best_match["title"],
            'matched_content': best_match["content"],
            'bm25_score': best_score,
        }
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
                SELECT id, query_text, result_title, bm25_score, created_at
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
