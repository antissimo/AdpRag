const askBtn        = document.getElementById("askBtn");
const questionInput = document.getElementById("question");
const answerDiv     = document.getElementById("answer");
const stepsDiv      = document.getElementById("steps");
const stepsDetails  = document.getElementById("stepsDetails");
const sourcesList   = document.getElementById("sourcesList");
const sourcesEmpty  = document.getElementById("sourcesEmpty");

// ── Modal elements ────────────────────────────────────────────────────────
const modal        = document.getElementById("modal");
const modalTitle   = document.getElementById("modalTitle");
const modalScore   = document.getElementById("modalScore");
const modalContent = document.getElementById("modalContent");
const modalClose   = document.getElementById("modalClose");

// ── State ─────────────────────────────────────────────────────────────────
let sourcesData = [];

// ── Ask (SSE streaming) ───────────────────────────────────────────────────
askBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!question) return alert("Please type a question!");

    // Reset UI
    askBtn.disabled     = true;
    askBtn.textContent  = "Loading...";
    answerDiv.innerHTML = "<em style='color:#aaa'>Thinking...</em>";
    stepsDiv.innerHTML  = "";
    stepsDetails.style.display = "none";
    sourcesList.innerHTML      = "";
    sourcesEmpty.style.display = "block";
    sourcesData = [];

    try {
        const response = await fetch("http://localhost:8000/ask/stream", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ question }),
        });

        if (!response.ok) {
            const err = await response.json();
            answerDiv.innerHTML = `<span style="color:red">Error ${response.status}: ${err.detail || "Unknown error"}</span>`;
            return;
        }

        // ── Read SSE stream ───────────────────────────────────────────────
        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let   buffer  = "";

        stepsDetails.style.display = "block";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // SSE messages are separated by double newline
            const parts = buffer.split("\n\n");
            buffer = parts.pop(); // keep incomplete last part

            for (const part of parts) {
                const line = part.trim();
                if (!line.startsWith("data:")) continue;

                try {
                    const payload = JSON.parse(line.slice(5).trim());
                    handleEvent(payload);
                } catch (e) {
                    console.warn("Failed to parse SSE payload:", line, e);
                }
            }
        }

    } catch (err) {
        answerDiv.innerHTML = `<span style="color:red">Error: ${err.message}</span>`;
    } finally {
        askBtn.disabled    = false;
        askBtn.textContent = "Ask";
    }
});

// ── Handle incoming SSE events ────────────────────────────────────────────
function handleEvent(payload) {
    if (payload.type === "step") {
        appendStep(payload.text);
    } else if (payload.type === "done") {
        renderAnswer(payload);
    }
}

// ── Append a single step line ─────────────────────────────────────────────
function appendStep(text) {
    const line = document.createElement("div");
    line.textContent = text;
    stepsDiv.appendChild(line);
    stepsDiv.scrollTop = stepsDiv.scrollHeight;
}

// ── Render final answer + sources ─────────────────────────────────────────
function renderAnswer(payload) {
    const formatted = payload.answer.replace(/\n/g, "<br>");
    answerDiv.innerHTML = `<strong>Answer:</strong><br>${formatted}`;

    if (payload.sources && payload.sources.length > 0) {
        sourcesEmpty.style.display = "none";
        sourcesData = payload.sources;
        renderSourceCards(payload.sources);
    }
}

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
    const src                = sourcesData[index];
    modalTitle.textContent   = src.document;
    modalScore.textContent   = `Relevance score: ${src.relevance_score ?? "—"}`;
    modalContent.textContent = src.full_text || src.chunk_preview;
    modal.style.display      = "flex";
}

function closeModal() {
    modal.style.display = "none";
}

modalClose.addEventListener("click", closeModal);
modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });