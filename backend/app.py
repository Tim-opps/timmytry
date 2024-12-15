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

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "34.67.248.245"),
    "user": os.getenv("DB_USER", "tps"),
    "password": os.getenv("DB_PASSWORD", "0423"),
    "database": os.getenv("DB_NAME", "fake_news_db")
}

# 模型及分词器路径
MODEL_PATH = os.getenv("MODEL_PATH", "FNCwithLSTM.h5")
WORD_INDEX_PATH = os.getenv("WORD_INDEX_PATH", "word_index.json")

# 全局变量
model = None
tokenizer = None

# 数据库连接池
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

        # 加载模型
        model = kr.models.load_model(MODEL_PATH)
        logging.info(f"LSTM 模型加载成功，耗时: {time.time() - start_time:.4f} 秒")

        # 加载分词器
        with open(WORD_INDEX_PATH, 'r', encoding='utf-8') as f:
            word_index = json.load(f)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
        tokenizer.word_index = word_index
        tokenizer.index_word = {index: word for word, index in word_index.items()}
        logging.info("分词器加载成功。")
    except Exception as e:
        logging.error(f"加载 LSTM 模型或分词器失败: {e}")

# 初始化模型和分词器
load_model_and_tokenizer()

# 文本预处理
def preprocess_texts(title):
    if tokenizer is None:
        raise ValueError("分词器尚未加载。")
    words = [word for word, flag in pseg.cut(title) if flag != 'x']
    title_tokenized = " ".join(words)
    x_test = tokenizer.texts_to_sequences([title_tokenized])
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

        # 查询数据库中的所有标题
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, title FROM cleaned_file")
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        # 找到最佳匹配项
        best_match = None
        best_similarity = 0

        for row in rows:
            db_title = row['title']
            similarity = np.random.rand()  # 模拟计算相似度逻辑
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = row

        if best_match is None or best_similarity < 0.5:
            return jsonify({'error': '没有找到足够相似的数据'}), 404

        # 使用 LSTM 模型进行预测
        probabilities = predict_category(input_title, best_match["title"])
        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]  # 中文分类
        category = categories[category_index]

        response = {
            'input_title': input_title,
            'matched_title': best_match["title"],
            'category': category,
            'probabilities': {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)}
        }

        # 保存历史记录
        connection = get_database_connection()
        if connection:
            try:
                cursor = connection.cursor()
                query = """
                    INSERT INTO history (query_text, result_title, result_category, created_at)
                    VALUES (%s, %s, %s, NOW())
                """
                cursor.execute(query, (input_title, best_match["title"], category))
                connection.commit()
                cursor.close()
                logging.info("历史记录已保存。")
            except Exception as e:
                logging.error(f"保存历史记录失败: {e}")
            finally:
                connection.close()

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
                SELECT id, query_text, result_title, result_category, created_at
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
