const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const answerDiv = document.getElementById("answer");
const sourcesDiv = document.getElementById("sources");
const stepsDiv = document.getElementById("steps");

askBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!question) return alert("Please type a question!");

    answerDiv.innerHTML = "Loading...";
    sourcesDiv.innerHTML = "";
    stepsDiv.innerHTML = "";

    try {
        const response = await fetch("http://localhost:8000/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await response.json();

        answerDiv.innerHTML = `<strong>Answer:</strong> ${data.answer}`;
        sourcesDiv.innerHTML = `<strong>Sources:</strong><br>` + data.sources.map(s => `${s.document} — ${s.chunk_preview}`).join("<br>");
        stepsDiv.innerHTML = `<strong>Steps:</strong><br>` + data.steps.join("<br>");
    } catch (err) {
        answerDiv.innerHTML = "Error: " + err.message;
    }
});