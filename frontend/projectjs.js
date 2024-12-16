// 監聽提交按鈕點擊事件
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
        const text = await response.text(); // 獲取原始響應
        let data;

        if (!response.ok) {
            throw new Error(`伺服器返回錯誤: ${response.status} ${response.statusText}`);
        }

        // 解析 JSON 資料
        try {
            data = JSON.parse(text);
        } catch (e) {
            console.error('解析 JSON 失敗，響應內容:', text);
            throw new Error('後端返回數據格式無效，無法解析成 JSON。');
        }

        // 隱藏加載動畫
        loadingIndicator.style.display = 'none';

        // 顯示結果或錯誤訊息
        if (data.error) {
            updateResultError(data.error);
        } else {
            updateResult(data, inputText);
        }
    } catch (error) {
        // 隱藏加載動畫
        loadingIndicator.style.display = 'none';
        console.error('請求失敗:', error.message);
        alert(`請求失敗，請稍後重試！錯誤訊息：${error.message}`);
    }
});

// 更新分析結果到頁面
function updateResult(data, inputText) {
    const resultSection = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    const resultTimestamp = document.getElementById('result-timestamp');

    // 準備資料
    const fakeProbability = data.probabilities && data.probabilities["不同意"] !== undefined 
        ? (data.probabilities["不同意"] * 100).toFixed(2) : '0.00';

    // 更新結果內容（只顯示真/假消息結果、輸入內容、匹配內容、假消息機率）
    resultText.innerHTML = `
        <strong>您的輸入：</strong> ${inputText}<br>
        <strong>分析結果：</strong> ${data.classification || '未知'}<br>
        <strong>假消息機率：</strong> ${fakeProbability}%<br>
        <strong>匹配標題：</strong> ${data.matched_title || '無匹配內容'}<br>
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

// 隨機抽取頁面內容
document.addEventListener("DOMContentLoaded", function () {
    const trendContainer = document.querySelector(".trend-fakenews");

    // 使用 PapaParse 讀取並解析 CSV 文件
    fetch("https://cors-anywhere.herokuapp.com/https://raw.githubusercontent.com/Tim-opps/timmytry/refs/heads/main/frontend/datacombined_1_processed.csv")
    .then(response => response.text())
    .then(data => {
            // 使用 PapaParse 解析 CSV
            const parsedData = Papa.parse(data, { header: true }).data;

            // 篩選 classification = 1 的資料
            const filteredNews = parsedData.filter(news => news.classification === "1");

            // 隨機選取 4 筆資料
            const randomNews = [];
            while (randomNews.length < 4 && filteredNews.length > 0) {
                const index = Math.floor(Math.random() * filteredNews.length);
                randomNews.push(filteredNews.splice(index, 1)[0]);
            }

            // 渲染資料到頁面
            if (randomNews.length === 0) {
                trendContainer.innerHTML = "<p>沒有符合的假消息資料。</p>";
                return;
            }

            randomNews.forEach(news => {
                const newsItem = document.createElement("div");
                newsItem.classList.add("news");

                // 標題
                const title = document.createElement("h4");
                title.textContent = news.title || "無標題";

                // 摘要內容
                const content = document.createElement("p");
                content.textContent = news.content ? `${news.content.substring(0, 100)}...` : "無內容摘要";

                // 假消息分類
                const classification = document.createElement("p");
                classification.innerHTML = `<strong>分類:</strong> 假消息`;

                newsItem.appendChild(title);
                newsItem.appendChild(content);
                newsItem.appendChild(classification);
                trendContainer.appendChild(newsItem);
            });
        })
        .catch(error => {
            console.error("Error loading or processing CSV file:", error);
            trendContainer.innerHTML = "<p>載入資料時發生錯誤，請稍後再試。</p>";
        });
});

