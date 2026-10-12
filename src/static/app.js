// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const clientId = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);
let ws = null;
let currentJobId = null;
let selectedFile = null;

// DOM refs
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const clearFileBtn = document.getElementById("clear-file");
const modelSelect = document.getElementById("model-select");
const numSpeakers = document.getElementById("num-speakers");
const enableDiarization = document.getElementById("enable-diarization");
const languageInput = document.getElementById("language");
const uploadBtn = document.getElementById("upload-btn");
const progressSection = document.getElementById("progress-section");
const progressBar = document.getElementById("progress-bar");
const progressStage = document.getElementById("progress-stage");
const progressMessage = document.getElementById("progress-message");
const transcriptSection = document.getElementById("transcript-section");
const transcriptMeta = document.getElementById("transcript-meta");
const transcriptContent = document.getElementById("transcript-content");
const downloadBtn = document.getElementById("download-btn");
const downloadAudioBtn = document.getElementById("download-audio-btn");
const historyList = document.getElementById("history-list");
const serverStatus = document.getElementById("server-status");

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/${clientId}`);

  ws.onopen = () => {
    serverStatus.textContent = "connected";
    serverStatus.classList.add("connected");
  };

  ws.onclose = () => {
    serverStatus.textContent = "disconnected";
    serverStatus.classList.remove("connected");
    setTimeout(connectWebSocket, 3000);
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleWsMessage(msg);
  };

  // Keep-alive ping
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send("ping");
  }, 30000);
}

function handleWsMessage(msg) {
  if (msg.type === "pong") return;

  if (msg.job_id && msg.job_id !== currentJobId) return;

  switch (msg.type) {
    case "progress":
      showProgress(msg);
      break;
    case "segment":
      appendSegment(msg);
      break;
    case "complete":
      onJobComplete(msg);
      break;
    case "error":
      onJobError(msg);
      break;
  }
}

// ---------------------------------------------------------------------------
// Progress & transcript display
// ---------------------------------------------------------------------------
const STAGE_PROGRESS = {
  loading_model: 5,
  transcribing: 15,
  transcription_done: 50,
  diarizing: 55,
  diarization_done: 90,
  diarization_skipped: 90,
  saving: 95,
};

function showProgress(msg) {
  progressSection.classList.remove("hidden");
  const pct = STAGE_PROGRESS[msg.stage] || 0;
  progressBar.classList.remove("indeterminate");
  if (msg.stage === "transcribing" || msg.stage === "diarizing") {
    progressBar.classList.add("indeterminate");
  } else {
    progressBar.style.width = pct + "%";
  }
  progressStage.textContent = stageLabel(msg.stage);
  if (msg.message) progressMessage.textContent = msg.message;
}

function stageLabel(stage) {
  const labels = {
    loading_model: "Loading model...",
    transcribing: "Transcribing...",
    transcription_done: "Transcription complete",
    diarizing: "Identifying speakers...",
    diarization_done: "Diarization complete",
    diarization_skipped: "Diarization skipped",
    saving: "Saving transcript...",
  };
  return labels[stage] || stage;
}

function appendSegment(msg) {
  transcriptSection.classList.remove("hidden");
  const line = `[${fmtTime(msg.start)} - ${fmtTime(msg.end)}] ${msg.text}\n`;
  transcriptContent.textContent += line;
  transcriptContent.scrollTop = transcriptContent.scrollHeight;
}

function onJobComplete(msg) {
  progressBar.classList.remove("indeterminate");
  progressBar.style.width = "100%";
  progressStage.textContent = "Complete!";
  progressMessage.textContent = msg.message || "";

  // Rebuild transcript with speaker labels
  transcriptContent.textContent = "";
  if (msg.segments) {
    for (const seg of msg.segments) {
      const speaker = seg.speaker ? `${seg.speaker}: ` : "";
      transcriptContent.textContent += `[${fmtTime(seg.start)} - ${fmtTime(seg.end)}] ${speaker}${seg.text}\n`;
    }
  }

  const durationStr = msg.duration ? fmtTime(msg.duration) : "";
  const speakersStr = msg.num_speakers ? `${msg.num_speakers} speakers` : "";
  transcriptMeta.textContent = [durationStr, speakersStr].filter(Boolean).join(" | ");

  transcriptSection.classList.remove("hidden");
  downloadBtn.classList.remove("hidden");
  downloadAudioBtn.classList.remove("hidden");
  uploadBtn.disabled = false;
  uploadBtn.textContent = "Upload & Transcribe";

  saveJobToHistory(currentJobId, "complete");
}

function onJobError(msg) {
  progressBar.classList.remove("indeterminate");
  progressBar.style.width = "0%";
  progressStage.textContent = "Error";
  progressMessage.textContent = msg.message || "Unknown error";
  uploadBtn.disabled = false;
  uploadBtn.textContent = "Upload & Transcribe";

  saveJobToHistory(currentJobId, "error");
}

function fmtTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) selectFile(fileInput.files[0]); });
clearFileBtn.addEventListener("click", clearFile);

function selectFile(file) {
  selectedFile = file;
  fileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  fileInfo.classList.remove("hidden");
  dropZone.classList.add("hidden");
  uploadBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  fileInfo.classList.add("hidden");
  dropZone.classList.remove("hidden");
  uploadBtn.disabled = true;
}

// ---------------------------------------------------------------------------
// Upload & start
// ---------------------------------------------------------------------------
uploadBtn.addEventListener("click", uploadAndStart);

async function uploadAndStart() {
  if (!selectedFile) return;

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  transcriptContent.textContent = "";
  transcriptSection.classList.add("hidden");
  progressSection.classList.remove("hidden");
  progressBar.style.width = "0%";
  progressStage.textContent = "Uploading file...";
  progressMessage.textContent = `${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB)`;

  try {
    // Upload
    const form = new FormData();
    form.append("file", selectedFile);
    let uploadRes;
    try {
      uploadRes = await fetch("/api/upload", { method: "POST", body: form });
    } catch (netErr) {
      throw new Error("Network error during upload - is the server running? (" + netErr.message + ")");
    }
    if (!uploadRes.ok) {
      const detail = await uploadRes.text().catch(() => `HTTP ${uploadRes.status}`);
      throw new Error("Upload rejected by server: " + detail);
    }
    const uploadData = await uploadRes.json();
    const job_id = uploadData.job_id;
    currentJobId = job_id;

    // Start processing
    progressStage.textContent = "Starting processing...";
    const model = modelSelect.value;
    const speakers = parseInt(numSpeakers.value) || 0;
    const lang = languageInput.value.trim();
    const diarize = enableDiarization.checked;
    const params = new URLSearchParams({ model, num_speakers: speakers, language: lang, client_id: clientId, diarize });
    let startRes;
    try {
      startRes = await fetch(`/api/jobs/${job_id}/start?${params}`, { method: "POST" });
    } catch (netErr) {
      throw new Error("Network error starting job (" + netErr.message + ")");
    }
    if (!startRes.ok) {
      const detail = await startRes.text().catch(() => `HTTP ${startRes.status}`);
      throw new Error("Server rejected start: " + detail);
    }

    uploadBtn.textContent = "Processing...";
    progressStage.textContent = "Queued...";
    progressMessage.textContent = "";

    saveJobToHistory(job_id, "processing");
  } catch (err) {
    console.error("Upload error:", err);
    progressStage.textContent = "Error";
    progressMessage.textContent = err.message;
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload & Transcribe";
  }
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------
downloadBtn.addEventListener("click", () => {
  if (currentJobId) window.open(`/api/jobs/${currentJobId}/download`, "_blank");
});
downloadAudioBtn.addEventListener("click", () => {
  if (currentJobId) window.open(`/api/jobs/${currentJobId}/download-audio`, "_blank");
});

// ---------------------------------------------------------------------------
// Job history (localStorage)
// ---------------------------------------------------------------------------
function loadHistory() {
  try { return JSON.parse(localStorage.getItem("ttt_history") || "[]"); } catch { return []; }
}

function saveJobToHistory(jobId, status) {
  const history = loadHistory();
  const existing = history.find((h) => h.id === jobId);
  if (existing) {
    existing.status = status;
    existing.updated = new Date().toISOString();
  } else {
    history.unshift({
      id: jobId,
      filename: selectedFile ? selectedFile.name : "unknown",
      status,
      created: new Date().toISOString(),
      updated: new Date().toISOString(),
    });
  }
  // Keep last 50
  localStorage.setItem("ttt_history", JSON.stringify(history.slice(0, 50)));
  renderHistory();
}

function renderHistory() {
  const history = loadHistory();
  if (!history.length) {
    historyList.innerHTML = '<p class="hint">No previous jobs.</p>';
    return;
  }
  historyList.innerHTML = history
    .map((h) => {
      const chipClass = h.status === "complete" ? "complete" : h.status === "error" ? "error" : "processing";
      return `<div class="history-item" data-job-id="${h.id}">
        <span>${h.filename}</span>
        <span class="status-chip ${chipClass}">${h.status}</span>
      </div>`;
    })
    .join("");

  historyList.querySelectorAll(".history-item").forEach((el) => {
    el.addEventListener("click", () => loadJob(el.dataset.jobId));
  });
}

async function loadJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) return;
    const job = await res.json();
    currentJobId = jobId;

    if (job.status === "complete" && job.has_transcript) {
      // Fetch and display the transcript via download endpoint
      const txtRes = await fetch(`/api/jobs/${jobId}/download`);
      if (txtRes.ok) {
        transcriptContent.textContent = await txtRes.text();
        transcriptSection.classList.remove("hidden");
        progressSection.classList.add("hidden");
        const durationStr = job.duration ? fmtTime(job.duration) : "";
        const speakersStr = job.num_speakers ? `${job.num_speakers} speakers` : "";
        transcriptMeta.textContent = [durationStr, speakersStr].filter(Boolean).join(" | ");
      }
    }
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  const diag = [];

  // Test GET
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    modelSelect.innerHTML = data.models
      .map((m) => `<option value="${m}" ${m === data.default ? "selected" : ""}>${m}</option>`)
      .join("");
    diag.push("GET OK");
  } catch (e) {
    modelSelect.innerHTML = '<option value="large-v3">large-v3</option>';
    diag.push("GET FAIL: " + e.message);
  }

  // Test POST (no file, just connectivity)
  try {
    const res = await fetch("/api/ping", { method: "POST" });
    if (res.ok) diag.push("POST OK");
    else diag.push("POST HTTP " + res.status);
  } catch (e) {
    diag.push("POST FAIL: " + e.message);
  }

  // Test POST with FormData (tiny blob, simulates file upload)
  try {
    const blob = new Blob(["test"], { type: "audio/wav" });
    const form = new FormData();
    form.append("file", blob, "diag_test.wav");
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (res.ok) diag.push("UPLOAD OK");
    else diag.push("UPLOAD HTTP " + res.status);
  } catch (e) {
    diag.push("UPLOAD FAIL: " + e.message);
  }

  // Check diarization availability
  try {
    const statusRes = await fetch("/api/status");
    if (statusRes.ok) {
      const status = await statusRes.json();
      if (!status.diarization_available) {
        enableDiarization.disabled = true;
        enableDiarization.parentElement.title = "HF token not configured on server";
      }
    }
  } catch { /* ignore */ }

  console.log("Diagnostics:", diag.join(" | "));
  serverStatus.textContent = diag.join(" | ");
  if (diag.every((d) => d.includes("OK"))) {
    serverStatus.classList.add("connected");
  }

  renderHistory();
  connectWebSocket();
}

init();
