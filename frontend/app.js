const titleByView = {
  dashboard: "Dashboard",
  analyze: "Analise tecnica",
  chat: "Chat tecnico",
  documents: "Documentos tecnicos",
};

const toast = document.querySelector("#toast");

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 3200);
}

function setView(view) {
  document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
  document.querySelector(`#view-${view}`).classList.add("active");
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
  document.querySelector(`[data-view="${view}"]`).classList.add("active");
  document.querySelector("#page-title").textContent = titleByView[view];

  if (view === "dashboard") loadHealth();
  if (view === "documents") loadDocuments();
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const message = data.detail || "Falha na requisicao.";
    throw new Error(typeof message === "string" ? message : pretty(message));
  }
  return data;
}

async function loadHealth() {
  const output = document.querySelector("#health-output");
  output.textContent = "Carregando...";
  try {
    const data = await apiJson("/health");
    const cards = document.querySelector("#health-cards");
    cards.innerHTML = `
      <div class="metric"><b>${data.status}</b><span>API</span></div>
      <div class="metric"><b>${data.ollama.ok ? "online" : "offline"}</b><span>Ollama</span></div>
      <div class="metric"><b>${data.index_loaded ? "carregado" : "lazy"}</b><span>Indice RAG</span></div>
    `;
    output.textContent = pretty(data);
  } catch (error) {
    output.textContent = String(error);
  }
}

async function analyzeById() {
  const output = document.querySelector("#analyze-output");
  output.textContent = "Processando...";
  try {
    const eventId = document.querySelector("#event-id").value.trim();
    const data = await apiJson("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: eventId, top_k_chunks: 3, similar_events_limit: 3 }),
    });
    output.textContent = pretty(data);
  } catch (error) {
    output.textContent = String(error);
  }
}

async function analyzeJson() {
  const output = document.querySelector("#analyze-output");
  output.textContent = "Processando...";
  try {
    const raw = document.querySelector("#event-json").value.trim();
    if (!raw) throw new Error("Informe um JSON.");
    const data = await apiJson("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event: JSON.parse(raw), top_k_chunks: 3, similar_events_limit: 3 }),
    });
    output.textContent = pretty(data);
  } catch (error) {
    output.textContent = String(error);
  }
}

async function sendChat() {
  const answer = document.querySelector("#chat-answer");
  const debug = document.querySelector("#chat-debug");
  const question = document.querySelector("#chat-question").value.trim();
  if (!question) {
    showToast("Digite uma pergunta.");
    return;
  }

  answer.textContent = "Consultando documentos e LLM...";
  debug.textContent = "";

  try {
    const data = await apiJson("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k_chunks: 4 }),
    });
    answer.textContent = data.answer;
    debug.textContent = pretty(data);
  } catch (error) {
    answer.textContent = String(error);
  }
}

async function loadDocuments() {
  const list = document.querySelector("#document-list");
  list.innerHTML = "<div class='document-row'>Carregando documentos...</div>";

  try {
    const data = await apiJson("/documents");
    list.innerHTML = "";

    if (!data.documents.length) {
      list.innerHTML = "<div class='document-row'>Nenhum PDF cadastrado.</div>";
      return;
    }

    data.documents.forEach((doc) => {
      const row = document.createElement("div");
      row.className = "document-row";
      const methods = (doc.methods || []).join(", ") || "sem metodo";
      row.innerHTML = `
        <div>
          <strong>${doc.document}</strong>
          <small>${doc.pages || 0} paginas | ${doc.image_pages || 0} com imagens | ${doc.ocr_pages || 0} com OCR | ${doc.ocr_unavailable_pages || 0} sem OCR disponivel | ${doc.characters || 0} caracteres | ${methods}</small>
        </div>
        <button class="danger" data-delete="${doc.document}">Deletar</button>
      `;
      list.appendChild(row);
    });
  } catch (error) {
    list.innerHTML = `<div class='document-row'>${String(error)}</div>`;
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  const fileInput = document.querySelector("#document-file");
  const overwrite = document.querySelector("#overwrite-document").checked;
  if (!fileInput.files.length) {
    showToast("Selecione um PDF.");
    return;
  }

  const form = new FormData();
  form.append("file", fileInput.files[0]);

  try {
    await apiJson(`/documents?overwrite=${overwrite}`, {
      method: "POST",
      body: form,
    });
    fileInput.value = "";
    showToast("Documento adicionado. Indice RAG sera recriado na proxima consulta.");
    await loadDocuments();
  } catch (error) {
    showToast(String(error));
  }
}

async function deleteDocument(filename) {
  if (!window.confirm(`Deletar ${filename}?`)) return;
  try {
    await apiJson(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    showToast("Documento removido. Indice RAG sera recriado na proxima consulta.");
    await loadDocuments();
  } catch (error) {
    showToast(String(error));
  }
}

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => setView(item.dataset.view));
});

document.querySelector("#refresh-health").addEventListener("click", loadHealth);
document.querySelector("#analyze-id").addEventListener("click", analyzeById);
document.querySelector("#analyze-json").addEventListener("click", analyzeJson);
document.querySelector("#send-chat").addEventListener("click", sendChat);
document.querySelector("#refresh-documents").addEventListener("click", loadDocuments);
document.querySelector("#upload-form").addEventListener("submit", uploadDocument);

document.querySelector(".chips").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-question]");
  if (!button) return;
  document.querySelector("#chat-question").value = button.dataset.question;
  sendChat();
});

document.querySelector("#document-list").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-delete]");
  if (!button) return;
  deleteDocument(button.dataset.delete);
});

loadHealth();
