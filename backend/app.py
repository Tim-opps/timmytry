import os
import json
import numpy as np
import jieba.posseg as pseg
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import mysql.connector
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

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "34.67.248.245"),
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

# 创建 history 表（如果不存在）
def create_history_table():
    connection = get_database_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = """
                CREATE TABLE IF NOT EXISTS history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    matched_title TEXT,
                    result_category VARCHAR(50),
                    fake_probability FLOAT,
                    real_probability FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            cursor.execute(query)
            connection.commit()
            logging.info("history 表已创建。")
        except Exception as e:
            logging.error(f"创建 history 表失败: {e}")
        finally:
            connection.close()

create_history_table()

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

# 分词函数
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']  # 返回分词后的列表

# 数据库匹配函数
def find_best_match(input_text):
    connection = get_database_connection()
    if not connection:
        raise RuntimeError("无法连接到数据库。")

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT id, title, content FROM cleaned_file LIMIT 200 ;"
        cursor.execute(query)
        records = cursor.fetchall()

        best_match = None
        best_score = 0

        for record in records:
            title_tokens = jieba_tokenizer(record['title'])
            content_tokens = jieba_tokenizer(record['content'])
            combined_text = " ".join(title_tokens + content_tokens)
            input_tokens = " ".join(jieba_tokenizer(input_text))

            # 简单匹配分数计算
            common_tokens = set(input_tokens.split()) & set(combined_text.split())
            score = len(common_tokens) / len(set(input_tokens.split()))

            if score > best_score and score >= SIMILARITY_THRESHOLD:
                best_score = score
                best_match = record

        return best_match, best_score

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
        step1_time = time.time()
        logging.info(f"解析请求数据耗时: {step1_time - start_time:.4f} 秒")

        # Step 2: 数据库匹配
        match_start_time = time.time()
        best_match, best_score = find_best_match(input_title)
        if not best_match:
            return jsonify({'error': '没有找到足够相似的数据'}), 404
        match_end_time = time.time()
        logging.info(f"数据库匹配耗时: {match_end_time - match_start_time:.4f} 秒")

        # Step 3: 使用 LSTM 模型进行预测
        model_start_time = time.time()
        probabilities = predict_category(input_title, best_match["title"])
        model_end_time = time.time()
        logging.info(f"LSTM 模型预测耗时: {model_end_time - model_start_time:.4f} 秒")

        # Step 4: 准备响应数据
        response_start_time = time.time()
        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]  # 中文分类
        category = categories[category_index]

        response = {
            'input_title': input_title,
            'matched_title': best_match["title"],
            'matched_content': best_match["content"],
            'match_score': best_score,
            'category': category,
            'probabilities': {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)}
        }
        response_end_time = time.time()
        logging.info(f"准备响应数据耗时: {response_end_time - response_start_time:.4f} 秒")

        # Step 5: 保存历史记录
        history_start_time = time.time()
        connection = get_database_connection()
        if connection:
            try:
                cursor = connection.cursor()
                query = """
                    INSERT INTO history (query_text, matched_title, result_category, fake_probability, real_probability)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    input_title,
                    best_match["title"],
                    category,
                    response['probabilities'].get("不同意", 0),
                    response['probabilities'].get("同意", 0)
                ))
                connection.commit()
                logging.info("历史记录已保存。")
            except Exception as e:
                logging.error(f"保存历史记录失败: {e}")
            finally:
                connection.close()
        history_end_time = time.time()
        logging.info(f"保存历史记录耗时: {history_end_time - history_start_time:.4f} 秒")

        # 总耗时记录
        total_time = time.time() - start_time
        logging.info(f"API 处理总时间: {total_time:.4f} 秒")
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
                SELECT id, query_text, matched_title, result_category, fake_probability, real_probability, created_at
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
        connection = get_database_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            query = "SELECT title, content FROM cleaned_file ORDER BY RAND() LIMIT 4"
            cursor.execute(query)
            news_data = cursor.fetchall()
            cursor.close()
            connection.close()
            return jsonify({'news': news_data})
        else:
            return jsonify({'error': 'Failed to connect to database'}), 500
    except Exception as e:
        logging.error(f"Error fetching random news: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
