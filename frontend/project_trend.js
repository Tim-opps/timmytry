document.addEventListener("DOMContentLoaded", function () {
    const trendContainer = document.querySelector(".trend-fakenews");

    // 加載動畫
    const loadingIndicator = document.createElement("div");
    loadingIndicator.textContent = "數據加載中...";
    loadingIndicator.style.textAlign = "center";
    loadingIndicator.style.fontSize = "1.2em";
    trendContainer.appendChild(loadingIndicator);

    // 錯誤提示函數
    function displayError(message) {
        trendContainer.innerHTML = `<p style="color: red; text-align: center;">錯誤: ${message}</p>`;
    }

    // 讀取並顯示新聞的函數
    function loadFakeNews() {
        loadingIndicator.style.display = "block";
        trendContainer.innerHTML = ""; // 清空內容

        fetch("datacombined_1_processed.csv")
            .then(response => {
                if (!response.ok) {
                    throw new Error(`無法加載新聞文件，HTTP 狀態碼: ${response.status}`);
                }
                return response.text();
            })
            .then(data => {
                // 使用 PapaParse 解析 CSV
                const parsedData = Papa.parse(data, { header: true, skipEmptyLines: true }).data;

                // 檢查 CSV 是否包含必要欄位
                if (!parsedData || !parsedData.length) {
                    throw new Error("新聞數據為空或格式不正確。");
                }

                // 過濾數據，避免空值
                const validNews = parsedData.filter(news => news.title && news.content);

                if (validNews.length === 0) {
                    throw new Error("新聞數據中沒有可顯示的內容。");
                }

                // 隨機選取 4 條新聞
                const randomNews = [];
                while (randomNews.length < 4 && validNews.length > 0) {
                    const index = Math.floor(Math.random() * validNews.length);
                    randomNews.push(validNews.splice(index, 1)[0]);
                }

                // 渲染新聞到頁面
                trendContainer.innerHTML = ""; // 清空內容
                randomNews.forEach(news => {
                    const newsItem = document.createElement("div");
                    newsItem.classList.add("news");

                    // 標題
                    const title = document.createElement("h4");
                    title.textContent = news.title || "無標題";
                    title.style.color = "#fbaf59";

                    // 內容摘要
                    const summary = document.createElement("p");
                    summary.textContent = news.content
                        ? news.content.substring(0, 100) + "..." // 取前 100 個字作為摘要
                        : "無內容摘要";

                    newsItem.appendChild(title);
                    newsItem.appendChild(summary);
                    trendContainer.appendChild(newsItem);
                });
            })
            .catch(error => {
                console.error("加載新聞失敗:", error);
                displayError(error.message || "加載新聞時發生未知錯誤，請稍後重試。");
            })
            .finally(() => {
                loadingIndicator.style.display = "none"; // 隱藏加載動畫
            });
    }

    // 初次加載新聞
    loadFakeNews();

    // 重新加載新聞按鈕
    const refreshButton = document.createElement("button");
    refreshButton.textContent = "重新加載新聞";
    refreshButton.style.margin = "20px auto";
    refreshButton.style.display = "block";
    document.body.appendChild(refreshButton);

    refreshButton.addEventListener("click", loadFakeNews);
});
