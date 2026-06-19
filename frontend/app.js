const form = document.querySelector("#uploadForm");
const imageInput = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const previewImage = document.querySelector("#previewImage");
const emptyPreview = document.querySelector("#emptyPreview");
const submitButton = document.querySelector("#submitButton");
const resetButton = document.querySelector("#resetButton");
const confInput = document.querySelector("#confInput");
const iouInput = document.querySelector("#iouInput");
const confValue = document.querySelector("#confValue");
const iouValue = document.querySelector("#iouValue");
const healthStatus = document.querySelector("#healthStatus");
const verdictTitle = document.querySelector("#verdictTitle");
const scoreChip = document.querySelector("#scoreChip");
const primaryClass = document.querySelector("#primaryClass");
const defectCount = document.querySelector("#defectCount");
const inferenceTime = document.querySelector("#inferenceTime");
const overlayImage = document.querySelector("#overlayImage");
const maskImage = document.querySelector("#maskImage");
const emptyResult = document.querySelector("#emptyResult");
const predictionRows = document.querySelector("#predictionRows");
const tabButtons = document.querySelectorAll(".tab-button");
const previewFrame = document.querySelector(".preview-frame");
const imageStage = document.querySelector(".image-stage");

let selectedFile = null;
let previewUrl = null;
let previewViewer = null;
let resultViewer = null;

function setHealth(state, text) {
  healthStatus.classList.remove("ok", "error");
  if (state) {
    healthStatus.classList.add(state);
  }
  healthStatus.querySelector("span:last-child").textContent = text;
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error("Model unavailable");
    }
    const data = await response.json();
    const classCount = Object.keys(data.classes || {}).length;
    setHealth("ok", `Model is ready`);
  } catch (error) {
    setHealth("error", "Model is not ready");
  }
}

function updateRangeLabels() {
  confValue.value = Number(confInput.value).toFixed(2);
  iouValue.value = Number(iouInput.value).toFixed(2);
}

function createImageViewer(frame, images) {
  const state = {
    scale: 1,
    x: 0,
    y: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    startX: 0,
    startY: 0,
  };

  const minScale = 1;
  const maxScale = 8;
  const zoomStep = 1.2;

  function hasImage() {
    return images.some((image) => image.classList.contains("active") && Boolean(image.src));
  }

  function applyTransform() {
    const transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
    images.forEach((image) => {
      image.style.transform = transform;
    });
    frame.classList.toggle("zoomed", state.scale > 1);
    frame.classList.toggle("has-image", hasImage());
  }

  function reset() {
    state.scale = 1;
    state.x = 0;
    state.y = 0;
    applyTransform();
  }

  function zoomAt(clientX, clientY, nextScale) {
    if (!hasImage()) {
      return;
    }

    const rect = frame.getBoundingClientRect();
    const pointX = clientX - rect.left;
    const pointY = clientY - rect.top;
    const clampedScale = Math.min(maxScale, Math.max(minScale, nextScale));
    const imageX = (pointX - state.x) / state.scale;
    const imageY = (pointY - state.y) / state.scale;

    state.x = pointX - imageX * clampedScale;
    state.y = pointY - imageY * clampedScale;
    state.scale = clampedScale;

    if (state.scale === minScale) {
      state.x = 0;
      state.y = 0;
    }

    applyTransform();
  }

  frame.addEventListener("wheel", (event) => {
    if (!hasImage()) {
      return;
    }

    event.preventDefault();
    const direction = event.deltaY < 0 ? zoomStep : 1 / zoomStep;
    zoomAt(event.clientX, event.clientY, state.scale * direction);
  }, { passive: false });

  frame.addEventListener("pointerdown", (event) => {
    if (!hasImage() || state.scale === minScale) {
      return;
    }

    state.isDragging = true;
    state.dragStartX = event.clientX;
    state.dragStartY = event.clientY;
    state.startX = state.x;
    state.startY = state.y;
    frame.setPointerCapture(event.pointerId);
    frame.classList.add("dragging");
  });

  frame.addEventListener("pointermove", (event) => {
    if (!state.isDragging) {
      return;
    }

    state.x = state.startX + event.clientX - state.dragStartX;
    state.y = state.startY + event.clientY - state.dragStartY;
    applyTransform();
  });

  frame.addEventListener("pointerup", (event) => {
    if (!state.isDragging) {
      return;
    }

    state.isDragging = false;
    frame.releasePointerCapture(event.pointerId);
    frame.classList.remove("dragging");
  });

  frame.addEventListener("pointercancel", () => {
    state.isDragging = false;
    frame.classList.remove("dragging");
  });

  frame.addEventListener("dblclick", reset);

  frame.querySelectorAll("[data-zoom]").forEach((button) => {
    button.addEventListener("pointerdown", (event) => event.stopPropagation());

    button.addEventListener("click", () => {
      const rect = frame.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      if (button.dataset.zoom === "in") {
        zoomAt(centerX, centerY, state.scale * zoomStep);
      } else if (button.dataset.zoom === "out") {
        zoomAt(centerX, centerY, state.scale / zoomStep);
      } else {
        reset();
      }
    });
  });

  reset();

  return {
    reset,
    refresh: applyTransform,
  };
}

function setPreview(file) {
  selectedFile = file;
  submitButton.disabled = !file;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  if (!file) {
    previewUrl = null;
    previewImage.removeAttribute("src");
    previewImage.classList.remove("active");
    emptyPreview.hidden = false;
    fileName.textContent = "JPG, PNG, BMP, WEBP";
    previewViewer.reset();
    return;
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.classList.add("active");
  emptyPreview.hidden = true;
  fileName.textContent = file.name;
  previewViewer.reset();
}

function resetResult() {
  verdictTitle.textContent = "No image available";
  scoreChip.textContent = "--";
  scoreChip.className = "score-chip";
  primaryClass.textContent = "--";
  defectCount.textContent = "--";
  inferenceTime.textContent = "--";
  overlayImage.removeAttribute("src");
  maskImage.removeAttribute("src");
  overlayImage.classList.remove("active");
  maskImage.classList.remove("active");
  emptyResult.hidden = false;
  predictionRows.innerHTML = '<tr><td colspan="4">--</td></tr>';
  setActiveTab("overlay");
  resultViewer.reset();
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading || !selectedFile;
  submitButton.textContent = isLoading ? "Analyzing" : "Inspect";
}

function verdictText(verdict) {
  if (verdict === "defect") {
    return "Welding Defect Detected";
  }
  if (verdict === "pass") {
    return "No Welding Defected";
  }
  return "No Detected Object";
}

function renderResult(data) {
  const summary = data.summary || {};
  const verdict = summary.verdict || "no_detection";
  const confidence = summary.confidence;

  verdictTitle.textContent = verdictText(verdict);
  scoreChip.className = `score-chip ${verdict === "defect" ? "defect" : verdict === "pass" ? "pass" : ""}`;
  scoreChip.textContent = confidence == null ? "--" : `${Math.round(confidence * 100)}%`;
  primaryClass.textContent = summary.primary_class || "--";
  defectCount.textContent = `${summary.defect_count ?? 0}/${summary.object_count ?? 0}`;
  inferenceTime.textContent = `${data.inference_ms ?? "--"} ms`;

  overlayImage.src = data.annotated_image;
  overlayImage.classList.add("active");
  emptyResult.hidden = true;

  if (data.mask_image) {
    maskImage.src = data.mask_image;
  } else {
    maskImage.removeAttribute("src");
  }

  renderRows(data.predictions || []);
  setActiveTab("overlay");
  resultViewer.reset();
}

function renderRows(predictions) {
  if (!predictions.length) {
    predictionRows.innerHTML = '<tr><td colspan="4">No Detection</td></tr>';
    return;
  }

  predictionRows.innerHTML = predictions.map((item) => {
    const bbox = (item.bbox || []).map((value) => Number(value).toFixed(0)).join(", ");
    const confidence = `${Math.round(item.confidence * 1000) / 10}%`;
    return `
      <tr>
        <td>${escapeHtml(item.class_name)}</td>
        <td>${confidence}</td>
        <td>${bbox}</td>
        <td>${item.mask_area_pixels || 0}</td>
      </tr>
    `;
  }).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setActiveTab(tabName) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });

  overlayImage.classList.toggle("active", tabName === "overlay" && Boolean(overlayImage.src));
  maskImage.classList.toggle("active", tabName === "mask" && Boolean(maskImage.src));
  emptyResult.hidden = Boolean(
    (tabName === "overlay" && overlayImage.src) ||
    (tabName === "mask" && maskImage.src)
  );
  resultViewer.refresh();
}

async function submitImage(event) {
  event.preventDefault();
  if (!selectedFile) {
    return;
  }

  const formData = new FormData(form);
  formData.set("file", selectedFile);

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    renderResult(data);
  } catch (error) {
    verdictTitle.textContent = "Can not analyzing image";
    scoreChip.textContent = "ERR";
    scoreChip.className = "score-chip defect";
    predictionRows.innerHTML = `<tr><td colspan="4">${escapeHtml(error.message)}</td></tr>`;
  } finally {
    setLoading(false);
  }
}

imageInput.addEventListener("change", (event) => {
  setPreview(event.target.files[0] || null);
  resetResult();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const file = event.dataTransfer.files[0];
  if (file) {
    imageInput.files = event.dataTransfer.files;
    setPreview(file);
    resetResult();
  }
});

resetButton.addEventListener("click", () => {
  form.reset();
  updateRangeLabels();
  setPreview(null);
  resetResult();
});

confInput.addEventListener("input", updateRangeLabels);
iouInput.addEventListener("input", updateRangeLabels);
form.addEventListener("submit", submitImage);

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

previewViewer = createImageViewer(previewFrame, [previewImage]);
resultViewer = createImageViewer(imageStage, [overlayImage, maskImage]);
updateRangeLabels();
resetResult();
loadHealth();

