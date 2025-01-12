function analyzeSentiment() {
    const review = document.getElementById("review").value;
    const result = document.getElementById("result");

    // Clear previous result
    result.textContent = "Analyzing...";

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ review: review })
    })
        .then(response => response.json())
        .then(data => {
            result.textContent = `Sentiment: ${data.sentiment}`;
        })
        .catch(error => {
            console.error("Error:", error);
            result.textContent = "Error analyzing sentiment.";
        });
}
