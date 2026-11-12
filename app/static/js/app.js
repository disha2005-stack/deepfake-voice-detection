const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const analyzeButton = document.querySelector("#analyze-button");
const recordButton = document.querySelector("#record-button");
const stopButton = document.querySelector("#stop-button");
const inputError = document.querySelector("#input-error");
const selectedAudio = document.querySelector("#selected-audio");
const audioPreview = document.querySelector("#audio-preview");
let activeFile = null;
let previewUrl = null;
let recorder = null;
let mediaStream = null;
let recordStartedAt = 0;
let durationTimer = null;

function showError(message) {
  inputError.textContent = message;
  inputError.hidden = !message;
}

function setAudioFile(file) {
  showError("");
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".wav")) {
    showError("Select a WAV file. Other formats are not accepted.");
    return;
  }
  if (file.size > 25 * 1024 * 1024) {
    showError("This file exceeds the 25 MB upload limit.");
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  activeFile = file;
  previewUrl = URL.createObjectURL(file);
  audioPreview.src = previewUrl;
  document.querySelector("#file-name").textContent = file.name;
  document.querySelector("#file-size").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  selectedAudio.hidden = false;
  analyzeButton.disabled = false;
}

function clearAudio() {
  activeFile = null;
  audioPreview.removeAttribute("src");
  audioPreview.load();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedAudio.hidden = true;
  analyzeButton.disabled = true;
  fileInput.value = "";
  showError("");
}

document.querySelector("#browse-button").addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => setAudioFile(fileInput.files[0]));
document.querySelector("#clear-file").addEventListener("click", clearAudio);
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
}
dropZone.addEventListener("drop", (event) => setAudioFile(event.dataTransfer.files[0]));

function updateRecordTime() {
  const seconds = Math.floor((Date.now() - recordStartedAt) / 1000);
  document.querySelector("#record-time").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

recordButton.addEventListener("click", async () => {
  showError("");
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/webm", "audio/ogg"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener("stop", async () => {
      clearInterval(durationTimer);
      mediaStream.getTracks().forEach((track) => track.stop());
      recordButton.disabled = false;
      stopButton.disabled = true;
      document.querySelector("#record-note").textContent = "Converting microphone audio to WAV…";
      try {
        const wav = await convertRecordingToWav(new Blob(chunks, { type: recorder.mimeType }));
        const file = new File([wav], `recording-${new Date().toISOString().replaceAll(":", "-")}.wav`, { type: "audio/wav" });
        setAudioFile(file);
        document.querySelector("#record-note").textContent = "Recording ready";
      } catch {
        showError("This browser could not decode its microphone recording. Try uploading a WAV file.");
        document.querySelector("#record-note").textContent = "Conversion failed";
      }
    }, { once: true });
    recorder.start();
    recordStartedAt = Date.now();
    updateRecordTime();
    durationTimer = setInterval(updateRecordTime, 500);
    recordButton.disabled = true;
    stopButton.disabled = false;
    document.querySelector("#record-note").textContent = "Recording from microphone";
  } catch {
    showError("Microphone access is unavailable. Allow microphone access or upload a WAV file.");
  }
});

stopButton.addEventListener("click", () => {
  if (recorder?.state === "recording") recorder.stop();
});

async function convertRecordingToWav(blob) {
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await blob.arrayBuffer());
    const channelData = Array.from({ length: decoded.numberOfChannels }, (_, index) => decoded.getChannelData(index));
    const samples = new Float32Array(decoded.length);
    channelData.forEach((channel) => {
      for (let index = 0; index < samples.length; index += 1) samples[index] += channel[index] / channelData.length;
    });
    const bytes = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(bytes);
    const write = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
    write(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    write(8, "WAVE");
    write(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, decoded.sampleRate, true);
    view.setUint32(28, decoded.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, "data");
    view.setUint32(40, samples.length * 2, true);
    for (let index = 0; index < samples.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, samples[index]));
      view.setInt16(44 + index * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
    }
    return new Blob([bytes], { type: "audio/wav" });
  } finally {
    await context.close();
  }
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function displayResult(result) {
  document.querySelector("#empty-result").hidden = true;
  document.querySelector("#loading-result").hidden = true;
  const panel = document.querySelector("#analysis-result");
  panel.hidden = false;
  const verdictRow = document.querySelector(".verdict-row");
  verdictRow.classList.toggle("real", result.prediction === "REAL");
  verdictRow.classList.toggle("fake", result.prediction === "FAKE");
  document.querySelector("#verdict").textContent = result.prediction;
  document.querySelector("#verdict-symbol").textContent = result.prediction === "REAL" ? "R" : "F";
  document.querySelector("#real-value").textContent = percent(result.real_probability);
  document.querySelector("#fake-value").textContent = percent(result.fake_probability);
  document.querySelector("#real-meter").style.width = percent(result.real_probability);
  document.querySelector("#fake-meter").style.width = percent(result.fake_probability);
  document.querySelector("#confidence-value").textContent = percent(result.confidence);
  document.querySelector("#duration-value").textContent = `${result.processing_seconds.toFixed(2)} s`;
  document.querySelector("#segment-count").textContent = result.segment_count;
  const list = document.querySelector("#segment-list");
  list.replaceChildren(...result.segments.map((segment) => {
    const card = document.createElement("article");
    card.className = "segment-card";
    const top = document.createElement("div");
    top.className = "segment-top";
    const label = document.createElement("strong");
    label.textContent = `SEGMENT ${String(segment.segment).padStart(2, "0")}`;
    const pill = document.createElement("span");
    pill.className = `segment-pill ${segment.prediction.toLowerCase()}`;
    pill.textContent = segment.prediction;
    top.append(label, pill);
    const meter = document.createElement("div");
    meter.className = `meter ${segment.prediction === "FAKE" ? "fake-meter" : ""}`;
    const fill = document.createElement("i");
    fill.style.width = percent(segment.fake_probability);
    meter.append(fill);
    const meta = document.createElement("div");
    meta.className = "segment-meta";
    meta.innerHTML = `<span>REAL ${percent(segment.real_probability)}</span><span>FAKE ${percent(segment.fake_probability)}</span>`;
    card.append(top, meter, meta);
    return card;
  }));
  document.querySelector("#spectrogram-empty").hidden = true;
  document.querySelector("#spectrogram-image").src = result.spectrogram;
  document.querySelector("#spectrogram-figure").hidden = false;
}

analyzeButton.addEventListener("click", async () => {
  if (!activeFile) return;
  showError("");
  analyzeButton.disabled = true;
  document.querySelector("#analysis-result").hidden = true;
  document.querySelector("#empty-result").hidden = true;
  document.querySelector("#loading-result").hidden = false;
  try {
    const body = new FormData();
    body.append("audio", activeFile, activeFile.name);
    const response = await fetch("/api/analyze", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis could not be completed.");
    displayResult(data);
    await loadHistory();
  } catch (error) {
    document.querySelector("#loading-result").hidden = true;
    document.querySelector("#empty-result").hidden = false;
    showError(error.message || "The server could not analyze this audio.");
  } finally {
    analyzeButton.disabled = !activeFile;
  }
});

async function loadHistory() {
  const response = await fetch("/api/history");
  const data = await response.json();
  const body = document.querySelector("#history-body");
  body.replaceChildren();
  if (!data.items.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 4;
    cell.className = "history-empty";
    cell.textContent = "No analyses saved on this device.";
  }
  for (const item of data.items) {
    const row = body.insertRow();
    row.insertCell().textContent = item.filename;
    const verdictCell = row.insertCell();
    verdictCell.className = `history-result ${item.prediction}`;
    verdictCell.textContent = item.prediction;
    row.insertCell().textContent = percent(item.confidence);
    row.insertCell().textContent = new Date(item.created_at).toLocaleString();
  }
  document.querySelector("#clear-history").disabled = data.items.length === 0;
}

document.querySelector("#clear-history").addEventListener("click", async () => {
  if (!window.confirm("Clear all saved detection history from this device?")) return;
  await fetch("/api/history", { method: "DELETE" });
  await loadHistory();
});

async function checkHealth() {
  const statusText = document.querySelector("#system-status");
  const statusDot = document.querySelector("#system-dot");
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Unavailable");
    statusText.textContent = "MODEL ONLINE · CPU/GPU";
    statusDot.classList.add("online");
  } catch {
    statusText.textContent = "MODEL OFFLINE";
    statusDot.classList.add("offline");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadHistory().catch(() => {});
});