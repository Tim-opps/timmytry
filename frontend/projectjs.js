// 初始化：監聽提交按鈕點擊事件
document.getElementById('submit-btn').addEventListener('click', async function () {
    const inputText = document.getElementById('inputtext').value.trim();

    // 驗證使用者是否輸入內容
    if (!inputText) {
        alert('請先輸入內容！');
        return;
    }

    // 顯示加載動畫
    const loadingIndicator = document.getElementById('loading');
    loadingIndicator.style.display = 'block';

    try {
        // 向後端發送請求
        const response = await fetch('https://timmytry.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: inputText }),
        });

        // 檢查請求是否成功
        if (!response.ok) {
            throw new Error(`伺服器返回錯誤: ${response.status} ${response.statusText}`);
        }

        // 解析後端返回的 JSON 資料
        const data = await response.json();

        // 隱藏加載動畫
        loadingIndicator.style.display = 'none';

        // 顯示結果或錯誤訊息
        if (data.error) {
            updateResultError(data.error);
        } else {
            updateResult(data);
        }
    } catch (error) {
        // 隱藏加載動畫
        loadingIndicator.style.display = 'none';
        console.error('請求失敗:', error.message);
        alert(`請求失敗，請稍後重試！錯誤訊息：${error.message}`);
    }
});

// 更新分析結果到頁面
function updateResult(data) {
    const resultSection = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    const resultTimestamp = document.getElementById('result-timestamp');

    // 更新結果內容
    resultText.innerHTML = `
        <strong>分析結果：</strong> ${data.category === 'fake' ? '假消息' : '真消息'}<br>
        <strong>假消息機率：</strong> ${(data.probabilities.fake * 100).toFixed(2)}%<br>
        <strong>真消息機率：</strong> ${(data.probabilities.real * 100).toFixed(2)}%<br>
        <strong>相關標題：</strong> ${data.matched_title || '無匹配標題'}<br>
        <strong>內容摘要：</strong> ${data.database_entry?.content || '無匹配內容'}
    `;

    // 顯示查詢時間戳
    const currentTime = new Date().toLocaleString();
    resultTimestamp.textContent = `查詢時間：${currentTime}`;

    // 顯示結果區域
    resultSection.style.display = 'block';

    // 平滑滾動到結果部分
    smoothScroll('#result');
}

// 更新錯誤訊息到頁面
function updateResultError(errorMessage) {
    const resultSection = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    const resultTimestamp = document.getElementById('result-timestamp');

    // 顯示錯誤訊息
    resultText.innerHTML = `<strong>錯誤：</strong>${errorMessage}`;

    // 顯示當前時間戳
    const currentTime = new Date().toLocaleString();
    resultTimestamp.textContent = `查詢時間：${currentTime}`;

    // 顯示結果區域
    resultSection.style.display = 'block';

    // 平滑滾動到結果部分
    smoothScroll('#result');
}

// 平滑滾動功能
function smoothScroll(target) {
    const element = document.querySelector(target);
    element.scrollIntoView({ behavior: 'smooth' });
}

// 重置結果區域的內容
function resetResult() {
    const resultSection = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    const resultTimestamp = document.getElementById('result-timestamp');

    // 清空內容
    resultText.innerHTML = '';
    resultTimestamp.textContent = '';

    // 隱藏結果區域
    resultSection.style.display = 'none';
}

// 監聽重置按鈕點擊事件
document.querySelector('button[type="reset"]').addEventListener('click', resetResult);



document.getElementById('submit').addEventListener('click', (event) => {
    // 防止表單預設的提交行為（刷新頁面）
    event.preventDefault();

    // 獲取輸入的文字內容
    const inputText = document.getElementById('inputtext').value.trim();

    // 檢查輸入是否為空
    if (!inputText) {
        alert("請輸入有疑慮的假消息！");
        return;
    }

    // 發送到後端的API進行處理
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ title: inputText }) // 傳遞輸入內容到後端
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            // 顯示錯誤信息
            alert(`錯誤：${data.error}`);
        } else {
            // 在這裡處理成功的回應結果
            console.log("預測結果：", data);
            alert(`結果：分類為 "${data.category}"\n相似標題：${data.matched_title}`);
        }
    })
    .catch(error => {
        console.error('發生錯誤:', error);
        alert("發送請求時發生錯誤，請稍後再試！");
    });
});


document.addEventListener("DOMContentLoaded", function () {
    const trendContainer = document.querySelector(".trend-fakenews");

    // 讀取並解析 CSV 文件
    fetch("datacombine_1.csv")
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(data => {
            const rows = data.split("\n").slice(1); // 去掉標題行
            const newsData = rows
                .filter(row => row.trim()) // 過濾空行
                .map(row => {
                    const [id, content, title, classification] = row.split(",").map(item => item.trim());
                    return { id, content, title, classification };
                });

            // 篩選 classification = 1 的資料
            const filteredNews = newsData.filter(news => news.classification === "1");

            // 隨機選取 4 筆資料
            const randomNews = [];
            while (randomNews.length < 4 && filteredNews.length > 0) {
                const index = Math.floor(Math.random() * filteredNews.length);
                randomNews.push(filteredNews.splice(index, 1)[0]);
            }

            // 將資料顯示在頁面上
            if (randomNews.length === 0) {
                trendContainer.innerHTML = "<p>沒有符合的假消息資料。</p>";
                return;
            }

            randomNews.forEach(news => {
                const newsItem = document.createElement("div");
                newsItem.classList.add("news-item");

                const title = document.createElement("h6");
                title.textContent = news.title;

                const content = document.createElement("p");
                content.textContent = news.content;

                newsItem.appendChild(title);
                newsItem.appendChild(content);
                trendContainer.appendChild(newsItem);
            });
        })
        .catch(error => {
            console.error("Error loading or processing CSV file:", error);
            trendContainer.innerHTML = "<p>載入資料時發生錯誤，請稍後再試。</p>";
        });
});


