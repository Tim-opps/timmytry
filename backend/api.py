import os
import mysql.connector
from flask import Flask, jsonify

app = Flask(__name__)

# 設定資料庫配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "34.67.248.245"),  # Cloud SQL 的外部 IP
    "user": os.getenv("DB_USER", "tps"),
    "password": os.getenv("DB_PASSWORD", "0423"),
    "database": os.getenv("DB_NAME", "fake_news_db")
}

@app.route('/api/random-news', methods=['GET'])
def random_news():
    try:
        # 連接資料庫
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # 查詢隨機4筆數據
        query = "SELECT Title, Content FROM news_table ORDER BY RAND() LIMIT 4"
        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(rows)

    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
