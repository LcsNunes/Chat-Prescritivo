const titleByView = {
  dashboard: "Dashboard",
  analyze: "Análise técnica",
  chat: "Chat técnico",
  documents: "Documentos técnicos",
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

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeText(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function markdownToHtml(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n{3,}/g, "\n\n");
}

function scoreLabel(score) {
  if (score === null || score === undefined) return "não informado";
  return `${(Number(score) * 100).toFixed(1)}%`;
}

function formatPeriod(period) {
  if (!period || !period.start || !period.end) return "sem periodo";
  return `${period.start.slice(0, 10)} a ${period.end.slice(0, 10)}`;
}

function parseAnswerSections(answer) {
  const sections = [];
  const regex = /(?:^|\n)\s*\*\*([^*\n]+)\*\*\s*/g;
  const matches = [...String(answer || "").matchAll(regex)];

  if (!matches.length) {
    return [{ title: "Resposta", body: answer || "" }];
  }

  matches.forEach((match, index) => {
    const start = match.index + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index : answer.length;
    sections.push({
      title: match[1].trim(),
      body: answer.slice(start, end).trim(),
    });
  });

  return sections.filter((section) => section.body);
}

function sectionHtml(sections, names) {
  const normalizedNames = names.map(normalizeText);
  const selected = sections.filter((section) =>
    normalizedNames.some((name) => normalizeText(section.title).includes(name))
  );

  if (!selected.length) return "";

  return selected
    .map((section) => `<strong>${escapeHtml(section.title)}</strong>\n${markdownToHtml(section.body)}`)
    .join("\n\n");
}

function renderAnalyzeResult(data) {
  const mapping = data.fault_mapping || {};
  const similar = data.similar_events || {};
  const chunks = data.retrieved_chunks || [];
  const guardrails = data.guardrails || {};
  const sections = parseAnswerSections(data.answer || "");

  const diagnosisHtml =
    sectionHtml(sections, [
      "tipo de falha",
      "evidências",
      "diagnóstico provável",
      "limitações",
    ]) || markdownToHtml(data.answer || "");

  const recommendationHtml =
    sectionHtml(sections, ["ações recomendadas", "cuidados de segurança"]) ||
    markdownToHtml(data.answer || "");

  const commonFaults = (similar.common_faults || [])
    .map(
      (item) =>
        `<li><strong>${escapeHtml(item.fault_normalized)}</strong><small>${escapeHtml(item.count)} ocorrência(s)</small></li>`
    )
    .join("");

  const examples = (similar.examples || [])
    .map(
      (item) => `
        <li>
          <strong>#${escapeHtml(item.id)} - ${escapeHtml(item.fault_normalized)}</strong>
          <small>${escapeHtml(item.created_at)} | distancia ${Number(item.similarity_distance).toFixed(3)}</small>
        </li>`
    )
    .join("");

  const chunkItems = chunks
    .map(
      (chunk) => `
        <li>
          <strong>${escapeHtml(chunk.document)} - página ${escapeHtml(chunk.page)} - score ${scoreLabel(chunk.score)}</strong>
          <small>${escapeHtml(chunk.text_preview)}</small>
        </li>`
    )
    .join("");

  const guardrailClass = guardrails.allowed ? "approved" : "blocked";
  const guardrailText = guardrails.allowed ? "Aprovado para LLM" : "Bloqueado por guardrail";

  return `
    <div class="result-grid">
      <article class="result-card">
        <h3>Falha e similaridade</h3>
        <div class="fact-grid">
          <div class="fact"><b>Falha original</b><span>${escapeHtml(mapping.fault_raw)}</span></div>
          <div class="fact"><b>Falha normalizada</b><span>${escapeHtml(mapping.fault_normalized)}</span></div>
          <div class="fact"><b>Classe canônica</b><span>${escapeHtml(mapping.display_name)}</span></div>
          <div class="fact"><b>Similaridade</b><span>${scoreLabel(mapping.score)} - ${escapeHtml(mapping.confidence)}</span></div>
        </div>
        <p><span class="pill ${guardrailClass}">${guardrailText}</span></p>
        <small>${escapeHtml(guardrails.message || "")}</small>
      </article>

      <article class="result-card">
        <h3>Históricos recentes</h3>
        <div class="fact-grid">
          <div class="fact"><b>Eventos similares</b><span>${escapeHtml(similar.count || 0)}</span></div>
          <div class="fact"><b>Periodo</b><span>${escapeHtml(formatPeriod(similar.period))}</span></div>
        </div>
        <h3 class="small-title">Falhas comuns</h3>
        <ul class="compact-list">${commonFaults || "<li>Nenhuma falha comum retornada.</li>"}</ul>
        <h3 class="small-title">Exemplos próximos</h3>
        <ul class="compact-list">${examples || "<li>Nenhum exemplo retornado.</li>"}</ul>
      </article>

      <article class="result-card wide">
        <h3>Recomendações</h3>
        <div class="answer-text">${recommendationHtml}</div>
      </article>

      <article class="result-card">
        <h3>Diagnóstico e evidências</h3>
        <div class="answer-text">${diagnosisHtml}</div>
      </article>

      <article class="result-card">
        <h3>Documentos recuperados</h3>
        <ul class="compact-list">${chunkItems || "<li>Nenhum chunk recuperado.</li>"}</ul>
      </article>

      <article class="result-card wide">
        <h3>JSON técnico</h3>
        <details class="raw-details">
          <summary>Mostrar resposta completa da API</summary>
          <pre class="code-output">${escapeHtml(pretty(data))}</pre>
        </details>
      </article>
    </div>
  `;
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    const message = data.detail || "Falha na requisição.";
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
      <div class="metric"><b>${data.index_loaded ? "carregado" : "lazy"}</b><span>Índice RAG</span></div>
    `;
    output.textContent = pretty(data);
  } catch (error) {
    output.textContent = String(error);
  }
}

async function analyzeById() {
  const output = document.querySelector("#analyze-output");
  output.className = "analysis-result empty";
  output.textContent = "Processando...";
  try {
    const eventId = document.querySelector("#event-id").value.trim();
    const data = await apiJson("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_id: eventId, top_k_chunks: 3, similar_events_limit: 3 }),
    });
    output.className = "analysis-result";
    output.innerHTML = renderAnalyzeResult(data);
  } catch (error) {
    output.className = "analysis-result empty";
    output.textContent = String(error);
  }
}

async function analyzeJson() {
  const output = document.querySelector("#analyze-output");
  output.className = "analysis-result empty";
  output.textContent = "Processando...";
  try {
    const event = readEventJson();
    const data = await apiJson("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event, top_k_chunks: 3, similar_events_limit: 3 }),
    });
    output.className = "analysis-result";
    output.innerHTML = renderAnalyzeResult(data);
  } catch (error) {
    output.className = "analysis-result empty";
    output.textContent = String(error);
  }
}

function readEventJson() {
  const raw = document.querySelector("#event-json").value.trim();
  if (!raw) throw new Error("Informe um JSON.");
  return JSON.parse(raw);
}

async function saveEvent() {
  const status = document.querySelector("#event-save-status");
  status.className = "save-status";
  status.textContent = "Gravando evento...";

  try {
    const event = readEventJson();
    const data = await apiJson("/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event }),
    });

    document.querySelector("#event-id").value = data.id;
    status.className = "save-status ok";
    const storageLabel = data.storage === "postgresql" ? "PostgreSQL" : "banner.csv";
    status.textContent =
      data.action === "updated"
        ? `Evento ${data.id} atualizado no ${storageLabel}.`
        : `Evento ${data.id} registrado no ${storageLabel}.`;

    if (data.ignored_fields?.length) {
      status.textContent += ` Campos ignorados: ${data.ignored_fields.join(", ")}.`;
    }
  } catch (error) {
    status.className = "save-status error";
    status.textContent = String(error);
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
          <small>${doc.pages || 0} páginas | ${doc.image_pages || 0} com imagens | ${doc.ocr_pages || 0} com OCR | ${doc.ocr_unavailable_pages || 0} sem OCR disponível | ${doc.characters || 0} caracteres | ${methods}</small>
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
    showToast("Documento adicionado. Índice RAG será recriado na próxima consulta.");
    await loadDocuments();
  } catch (error) {
    showToast(String(error));
  }
}

async function deleteDocument(filename) {
  if (!window.confirm(`Deletar ${filename}?`)) return;
  try {
    await apiJson(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    showToast("Documento removido. Índice RAG será recriado na próxima consulta.");
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
document.querySelector("#save-event").addEventListener("click", saveEvent);
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
