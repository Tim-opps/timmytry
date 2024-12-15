import os
import json
import numpy as np
import jieba.posseg as pseg
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import mysql.connector
from rank_bm25 import BM25Okapi
import time
import keras as kr
import tensorflow as tf

# 初始化日志记录
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 全局参数
SIMILARITY_THRESHOLD = 1.2  # BM25 相似度阈值
MAX_SEQUENCE_LENGTH = 20    # 模型输入的最大序列长度

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "35.185.148.251"),
    "user": os.getenv("DB_USER", "tps"),
    "password": os.getenv("DB_PASSWORD", "0423"),
    "database": os.getenv("DB_NAME", "fake_news_db")
}

# 全局变量
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

# 初始化 LSTM
load_model_and_tokenizer()

# 分词器
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']

# 动态获取BM25评分
def get_best_match_bm25(input_title, batch_size=100):
    connection = get_database_connection()
    if not connection:
        raise RuntimeError("无法连接到数据库。")
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, title, content FROM cleaned_file")
        best_score = 0
        best_match = None

        # 动态分块加载文档并计算BM25评分
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            # 提取文档标题和内容
            corpus = [jieba_tokenizer(row["title"] + " " + row["content"]) for row in rows]
            bm25 = BM25Okapi(corpus)
            
            # 计算BM25分数
            input_tokens = jieba_tokenizer(input_title)
            scores = bm25.get_scores(input_tokens)
            max_score_index = np.argmax(scores)
            
            if scores[max_score_index] > best_score and scores[max_score_index] >= SIMILARITY_THRESHOLD:
                best_score = scores[max_score_index]
                best_match = rows[max_score_index]

        cursor.close()
        return best_match, best_score
    except Exception as e:
        logging.error(f"BM25 计算失败: {e}")
        return None, 0
    finally:
        connection.close()

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
        logging.info("收到 /predict 请求")
        start_time = time.time()
        data = request.json
        logging.info(f"请求数据: {data}")
        
        input_title = data.get('title', '').strip()
        if not input_title:
            return jsonify({'error': '需要提供标题'}), 400

        # 记录BM25匹配日志
        best_doc_id, best_score = get_best_match_bm25(input_title)
        logging.info(f"BM25 匹配结果: Doc ID={best_doc_id}, 分数={best_score}")

        if not best_doc_id:
            return jsonify({'error': '没有找到相似数据'}), 404

        # 查询数据库
        best_match = get_database_record(best_doc_id)
        logging.info(f"数据库查询结果: {best_match}")

        if not best_match:
            return jsonify({'error': '无法获取匹配数据'}), 500

        # 模型预测
        probabilities = predict_category(input_title, best_match["title"])
        logging.info(f"模型预测结果: {probabilities}")

        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]
        category = categories[category_index]

        response = {
            'input_title': input_title,
            'matched_title': best_match["title"],
            'matched_content': best_match["content"],
            'bm25_score': best_score,
            'category': category,
            'probabilities': {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)}
        }

        # 保存历史记录
        connection = get_database_connection()
        if connection:
            try:
                cursor = connection.cursor()
                query = """
                    INSERT INTO history (query_text, result_title, bm25_score, result_category, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                """
                cursor.execute(query, (input_title, best_match["title"], best_score, category))
                connection.commit()
                cursor.close()
                logging.info("历史记录已保存")
            except Exception as e:
                logging.error(f"历史记录保存失败: {e}")
            finally:
                connection.close()

        logging.info(f"API 处理总时间: {time.time() - start_time:.4f} 秒")
        return jsonify(response)
    except Exception as e:
        logging.error(f"错误发生: {e}")
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


@app.route('/get-random-news', methods=['GET'])
def get_random_news():
    try:
        # 建立資料庫連線
        connection = get_database_connection()
        if not connection:
            return jsonify({'error': '無法連線到資料庫'}), 500
        
        # 隨機選取 4 筆資料
        cursor = connection.cursor(dictionary=True)
        query = "SELECT title, content FROM cleaned_file ORDER BY RAND() LIMIT 4"
        cursor.execute(query)
        news_data = cursor.fetchall()

        # 關閉連線
        cursor.close()
        connection.close()

        # 返回 JSON 結果
        return jsonify({'news': news_data}), 200

    except Exception as e:
        logging.error(f"取得隨機新聞時發生錯誤: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
