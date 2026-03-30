const askBtn       = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const answerDiv    = document.getElementById("answer");
const stepsDiv     = document.getElementById("steps");
const stepsDetails = document.getElementById("stepsDetails");
const sourcesList  = document.getElementById("sourcesList");
const sourcesEmpty = document.getElementById("sourcesEmpty");

// ── Modal elements ────────────────────────────────────────────────────────
const modal        = document.getElementById("modal");
const modalTitle   = document.getElementById("modalTitle");
const modalScore   = document.getElementById("modalScore");
const modalContent = document.getElementById("modalContent");
const modalClose   = document.getElementById("modalClose");

// ── Source cards storage (full chunk text) ────────────────────────────────
let sourcesData = [];

// ── Ask ───────────────────────────────────────────────────────────────────
askBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!question) return alert("Please type a question!");

    askBtn.disabled = true;
    askBtn.textContent = "Loading...";
    answerDiv.innerHTML = "<em style='color:#aaa'>Thinking...</em>";
    sourcesList.innerHTML = "";
    sourcesEmpty.style.display = "block";
    stepsDetails.style.display = "none";
    sourcesData = [];

    try {
        const response = await fetch("http://localhost:8000/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();

        if (!response.ok) {
            answerDiv.innerHTML = `<span style="color:red">Error ${response.status}: ${data.detail || "Unknown error"}</span>`;
            return;
        }

        // ── Answer ────────────────────────────────────────────────────────
        answerDiv.innerHTML = `<strong>Answer:</strong><br>${data.answer}`;

        // ── Steps ─────────────────────────────────────────────────────────
        if (data.steps && data.steps.length > 0) {
            stepsDiv.innerHTML = data.steps.join("<br>");
            stepsDetails.style.display = "block";
        }

        // ── Sources ───────────────────────────────────────────────────────
        if (data.sources && data.sources.length > 0) {
            sourcesEmpty.style.display = "none";
            sourcesData = data.sources;
            renderSourceCards(data.sources);
        }

    } catch (err) {
        answerDiv.innerHTML = `<span style="color:red">Error: ${err.message}</span>`;
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = "Ask";
    }
});

// ── Render source cards ───────────────────────────────────────────────────
function renderSourceCards(sources) {
    sourcesList.innerHTML = "";

    sources.forEach((src, index) => {
        const card = document.createElement("div");
        card.className = "source-card";
        card.innerHTML = `
            <div class="source-name" title="${src.document}">📄 ${src.document}</div>
            <div class="source-preview">${src.chunk_preview}</div>
        `;
        card.addEventListener("click", () => openModal(index));
        sourcesList.appendChild(card);
    });
}

// ── Modal ─────────────────────────────────────────────────────────────────
function openModal(index) {
    const src = sourcesData[index];
    modalTitle.textContent = src.document;
    //modalScore.textContent = `Relevance score: ${src.relevance_score ?? "—"}`;
    // chunk_preview is the full text passed from API (120 chars preview)
    // show full content if available, otherwise preview
    modalContent.textContent = src.full_text || src.chunk_preview;
    modal.style.display = "flex";
}

function closeModal() {
    modal.style.display = "none";
}

modalClose.addEventListener("click", closeModal);
modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
});
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
});