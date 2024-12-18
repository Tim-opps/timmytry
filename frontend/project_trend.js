document.addEventListener("DOMContentLoaded", function () {
    const trendContainer = document.querySelector(".trend-fakenews");
    const modal = document.getElementById("modal");
    const modalTitle = document.getElementById("modal-title");
    const modalBody = document.getElementById("modal-body");
    const closeModal = document.getElementById("close-modal");

    // 加載動畫
    const loadingIndicator = document.createElement("div");
    loadingIndicator.textContent = "數據加載中...";
    loadingIndicator.style.textAlign = "center";
    loadingIndicator.style.fontSize = "1.2em";
    trendContainer.appendChild(loadingIndicator);

    function loadFakeNews() {
        loadingIndicator.style.display = "block";
        trendContainer.innerHTML = ""; // 清空內容

        fetch("datacombined_1_processed.csv")
            .then(response => response.text())
            .then(data => {
                const parsedData = Papa.parse(data, { header: true, skipEmptyLines: true }).data;

                const validNews = parsedData.filter(news => news.title && news.content);

                const randomNews = [];
                while (randomNews.length < 4 && validNews.length > 0) {
                    const index = Math.floor(Math.random() * validNews.length);
                    randomNews.push(validNews.splice(index, 1)[0]);
                }

                trendContainer.innerHTML = ""; // 清空內容
                randomNews.forEach(news => {
                    const newsItem = document.createElement("div");
                    newsItem.classList.add("news");

                    const title = document.createElement("h4");
                    title.textContent = news.title || "無標題";
                    title.style.color = "#fbaf59";
                    title.style.cursor = "pointer"; // 可點擊

                    const summary = document.createElement("p");
                    summary.textContent = news.content
                        ? news.content.substring(0, 100) + "..." 
                        : "無內容摘要";

                    // 點擊標題顯示詳細內容
                    title.addEventListener("click", function () {
                        modalTitle.textContent = news.title || "無標題";
                        modalBody.textContent = news.content || "無詳細內容";
                        modal.style.display = "block";
                    });

                    newsItem.appendChild(title);
                    newsItem.appendChild(summary);
                    trendContainer.appendChild(newsItem);
                });
            })
            .catch(error => {
                console.error("加載新聞失敗:", error);
                trendContainer.innerHTML = "<p>加載新聞時發生錯誤，請稍後再試。</p>";
            })
            .finally(() => {
                loadingIndicator.style.display = "none";
            });
    }

    // 關閉彈出視窗
    closeModal.addEventListener("click", function () {
        modal.style.display = "none";
    });

        window.addEventListener("click", function (event) {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });

    // 初次加載新聞
    loadFakeNews();

   
});
