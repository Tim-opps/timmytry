import os
import json
import numpy as np
import pandas as pd
import jieba.posseg as pseg
from flask import Flask, request, jsonify,render_template
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
MAX_SEQUENCE_LENGTH = 20    # 模型输入的最大序列长度
SIMILARITY_THRESHOLD = 0.5  # 数据库匹配的最低分数
CSV_FILE = "datacombined_1_tokenized.csv"  # 分词后的 CSV 文件路径
HISTORY_CSV = "history.csv"   # 历史记录 CSV 文件路径

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
        logging.info("分词器加载成功。")
    except Exception as e:
        logging.error(f"加载 LSTM 模型或分词器失败: {e}")

load_model_and_tokenizer()

# 加载 CSV 数据
def load_csv_data():
    global data
    try:
        data = pd.read_csv(CSV_FILE)
        logging.info(f"分词后的 CSV 文件已加载，共 {len(data)} 条记录。")
    except Exception as e:
        logging.error(f"加载 CSV 文件失败: {e}")

load_csv_data()

# 分词函数
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return [word for word, flag in words if flag != 'x']  # 返回分词后的列表

# CSV 匹配函数
def find_best_match(input_text):
    best_match = None
    best_score = 0

    for _, row in data.iterrows():
        title_tokens = row['title'].split()
        content_tokens = row['content'].split()
        combined_text = set(title_tokens + content_tokens)
        input_tokens = set(jieba_tokenizer(input_text))

        # 简单匹配分数计算
        common_tokens = input_tokens & combined_text
        score = len(common_tokens) / len(input_tokens)

        if score > best_score and score >= SIMILARITY_THRESHOLD:
            best_score = score
            best_match = row

    return best_match, best_score

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

# 保存历史记录到 CSV
def save_history(input_title, matched_title, category, probabilities):
    try:
        history_data = {
            "query_text": input_title,
            "matched_title": matched_title,
            "result_category": category,
            "fake_probability": probabilities.get("不同意", 0),
            "real_probability": probabilities.get("同意", 0),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        if not os.path.exists(HISTORY_CSV):
            pd.DataFrame([history_data]).to_csv(HISTORY_CSV, index=False)
        else:
            pd.DataFrame([history_data]).to_csv(HISTORY_CSV, mode='a', header=False, index=False)
        logging.info("历史记录已保存。")
    except Exception as e:
        logging.error(f"保存历史记录失败: {e}")

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
        probabilities = predict_category(input_title, best_match["title"])
        model_end_time = time.time()
        logging.info(f"LSTM 模型预测耗时: {model_end_time - model_start_time:.4f} 秒")

        # Step 4: 准备响应数据
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

        # 保存历史记录
        save_history(input_title, best_match["title"], category, response['probabilities'])

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
        if not os.path.exists(HISTORY_CSV):
            return jsonify({'history': []})

        history_data = pd.read_csv(HISTORY_CSV).to_dict(orient='records')
        return jsonify({'history': history_data})
    except Exception as e:
        logging.error(f"获取历史记录失败: {e}")
        return jsonify({'error': str(e)}), 500


from flask import Flask, render_template
import pandas as pd

app = Flask(
    __name__,
    template_folder="../frontend/templates",  # 指定 HTML 模板文件夾
    static_folder="../frontend/static"        # 指定靜態資源文件夾
)

# 隨機抽取頁面路由
@app.route('/random')
def random_page():
    # 讀取 CSV 文件
    data = pd.read_csv('datacombined_1.csv')

    # 隨機抽取 4 筆資料
    sampled_data = data.sample(n=4).to_dict(orient='records')

    # 渲染 random.html，並將數據傳遞給模板
    return render_template('random.html', data=sampled_data)

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
