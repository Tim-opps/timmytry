import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import jieba.posseg as pseg
from rank_bm25 import BM25Okapi
import keras as kr
import tensorflow as tf
import logging
import time

# 初始化日志记录
logging.basicConfig(level=logging.INFO)

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

# 全局参数
BM25_FILE_PATH = "bm25_data.json"  # 本地存储的 BM25 数据
BM25_CHUNK_SIZE = 1000  # 每块加载的文档数量
SIMILARITY_THRESHOLD = 1.2  # BM25 相似度阈值
MAX_SEQUENCE_LENGTH = 20    # 模型输入的最大序列长度

# 全局变量
bm25 = None
corpus_chunks = []  # 存储分块后的文档
doc_ids_chunks = []  # 对应的文档ID分块
model = None
tokenizer = None

# 加载分词器
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != "x"]

# 加载 LSTM 模型和分词器
def load_model_and_tokenizer():
    global model, tokenizer
    try:
        start_time = time.time()
        model_path = os.getenv("MODEL_PATH", "FNCwithLSTM.h5")
        word_index_path = os.getenv("WORD_INDEX_PATH", "word_index.json")

        # 加载模型
        model = kr.models.load_model(model_path)
        logging.info(f"LSTM 模型加载成功，耗时 {time.time() - start_time:.2f} 秒")

        # 加载分词器
        with open(word_index_path, "r", encoding="utf-8") as f:
            word_index = json.load(f)
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
        tokenizer.word_index = word_index
        tokenizer.index_word = {index: word for word, index in word_index.items()}
        logging.info("分词器加载成功")
    except Exception as e:
        logging.error(f"加载 LSTM 模型或分词器失败: {e}")

# 分块加载 BM25 数据
def load_bm25_chunk(chunk_index):
    try:
        with open(BM25_FILE_PATH, "r", encoding="utf-8") as f:
            bm25_data = json.load(f)
        start = chunk_index * BM25_CHUNK_SIZE
        end = start + BM25_CHUNK_SIZE
        corpus_chunk = bm25_data["corpus"][start:end]
        doc_ids_chunk = bm25_data["doc_ids"][start:end]
        bm25_chunk = BM25Okapi(corpus_chunk)
        logging.info(f"加载 BM25 数据块 {chunk_index}（范围 {start}-{end}），文档数量: {len(corpus_chunk)}")
        return bm25_chunk, corpus_chunk, doc_ids_chunk
    except Exception as e:
        logging.error(f"加载 BM25 数据块失败: {e}")
        return None, None, None

# 获取最佳匹配项（BM25分块）
def get_best_match_bm25(input_text):
    global bm25, corpus, doc_ids
    if bm25 is None:
        raise RuntimeError("BM25 模型尚未加载。")

    input_tokens = jieba_tokenizer(input_text)
    best_doc_id = None
    best_score = 0

    chunk_index = 0
    while True:
        bm25_chunk, _, doc_ids_chunk = load_bm25_chunk(chunk_index)
        if not bm25_chunk:
            break
        scores = bm25_chunk.get_scores(input_tokens)
        local_best_index = np.argmax(scores)
        local_best_score = scores[local_best_index]

        if local_best_score > best_score and local_best_score >= SIMILARITY_THRESHOLD:
            best_score = local_best_score
            best_doc_id = doc_ids_chunk[local_best_index]

        chunk_index += 1

    return best_doc_id, best_score

# 文本预处理
def preprocess_texts(title):
    if tokenizer is None:
        raise ValueError("分词器尚未加载")
    title_tokenized = jieba_tokenizer(title)
    x_test = tokenizer.texts_to_sequences([" ".join(title_tokenized)])
    x_test = kr.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
    return x_test

# 模型预测
def predict_category(input_title, database_title):
    if model is None:
        raise ValueError("LSTM 模型尚未加载")
    input_processed = preprocess_texts(input_title)
    db_processed = preprocess_texts(database_title)
    predictions = model.predict([input_processed, db_processed])
    return predictions

# API 路由：预测
@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()
        data = request.json
        logging.info(f"收到的请求数据: {data}")
        input_title = data.get("title", "").strip()

        if not input_title:
            return jsonify({"error": "需要提供标题"}), 400

        if len(input_title) < 3:
            return jsonify({"error": "标题过短"}), 400

        # 使用分块 BM25 获取最佳匹配
        best_doc_id, best_score = get_best_match_bm25(input_title)
        if not best_doc_id:
            return jsonify({"error": "没有找到足够相似的数据"}), 404

        # 假设从数据库中获取匹配的文档标题
        best_match = {"id": best_doc_id, "title": "示例标题", "content": "示例内容"}  # 模拟数据

        # 使用 LSTM 模型进行预测
        probabilities = predict_category(input_title, best_match["title"])
        category_index = np.argmax(probabilities)
        categories = ["無關", "同意", "不同意"]
        category = categories[category_index]

        response = {
            "input_title": input_title,
            "matched_title": best_match["title"],
            "matched_content": best_match["content"],
            "bm25_score": best_score,
            "category": category,
            "probabilities": {cat: float(probabilities[0][i]) for i, cat in enumerate(categories)},
        }

        logging.info(f"API 处理总时间: {time.time() - start_time:.4f} 秒")
        return jsonify(response)

    except Exception as e:
        logging.error(f"发生错误: {e}")
        return jsonify({"error": str(e)}), 500

# API 路由：获取历史记录
@app.route("/history", methods=["GET"])
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
            return jsonify({"history": history_data})
        else:
            return jsonify({"error": "无法连接到数据库"}), 500
    except Exception as e:
        logging.error(f"获取历史记录失败: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
