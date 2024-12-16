document.addEventListener("DOMContentLoaded", function () {
    const trendContainer = document.querySelector(".trend-fakenews");
    const refreshButton = document.createElement("button");
    refreshButton.textContent = "重新加載新聞";
    refreshButton.style.margin = "20px auto";
    refreshButton.style.display = "block";
    document.body.appendChild(refreshButton); // 動態添加按鈕

    const loadingIndicator = document.createElement("div");
    loadingIndicator.textContent = "數據加載中...";
    loadingIndicator.style.display = "none";
    loadingIndicator.style.textAlign = "center";
    loadingIndicator.style.fontSize = "1.2em";
    document.body.insertBefore(loadingIndicator, trendContainer);

    // 讀取和渲染假新聞的函數
    function loadFakeNews() {
        loadingIndicator.style.display = "block"; // 顯示加載動畫
        trendContainer.innerHTML = ""; // 清空之前的內容

        fetch("datacombined_1_processed.csv") // CSV 文件位置
            .then(response => response.text())
            .then(data => {
                const parsedData = Papa.parse(data, { header: true }).data;

                // 篩選 classification = 1 的新聞
                const filteredNews = parsedData.filter(news => news.classification === "1");

                // 隨機選取 4 條新聞
                const randomNews = [];
                while (randomNews.length < 4 && filteredNews.length > 0) {
                    const index = Math.floor(Math.random() * filteredNews.length);
                    randomNews.push(filteredNews.splice(index, 1)[0]);
                }

                // 渲染新聞到頁面
                if (randomNews.length === 0) {
                    trendContainer.innerHTML = "<p>沒有符合的假新聞。</p>";
                    return;
                }

                randomNews.forEach(news => {
                    const newsItem = document.createElement("div");
                    newsItem.classList.add("news");

                    const title = document.createElement("h4");
                    title.textContent = news.tokenized_title || "無標題";
                    title.style.cursor = "pointer"; // 鼠標懸停效果
                    title.style.color = "#fbaf59";

                    const summary = document.createElement("p");
                    summary.textContent = news.tokenized_content
                        ? news.tokenized_content.substring(0, 50) + "..."
                        : "無內容摘要";

                    const classification = document.createElement("p");
                    classification.innerHTML = `<strong>分類:</strong> 假新聞`;

                    // 點擊標題顯示完整內容
                    title.addEventListener("click", function () {
                        alert(`完整內容：\n${news.tokenized_content || "無內容"}`);
                    });

                    newsItem.appendChild(title);
                    newsItem.appendChild(summary);
                    newsItem.appendChild(classification);
                    trendContainer.appendChild(newsItem);
                });
            })
            .catch(error => {
                console.error("加載新聞失敗:", error);
                trendContainer.innerHTML = "<p>加載新聞時發生錯誤，請稍後再試。</p>";
            })
            .finally(() => {
                loadingIndicator.style.display = "none"; // 隱藏加載動畫
            });
    }

    // 首次加載新聞
    loadFakeNews();

    // 重新加載新聞
    refreshButton.addEventListener("click", loadFakeNews);
});
