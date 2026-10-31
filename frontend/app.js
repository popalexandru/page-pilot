const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const buttonLabel = document.querySelector(".button-label");
const resultCard = document.querySelector("#result");
const answerStatus = document.querySelector("#answer-status");
const answerText = document.querySelector("#answer-text");
const sourcePages = document.querySelector("#source-pages");
const retrievedPages = document.querySelector("#retrieved-pages");
const message = document.querySelector("#message");
const serverStatus = document.querySelector("#server-status");
const manualFile = document.querySelector("#manual-file");
const manualName = document.querySelector("#manual-name");
const manualMeta = document.querySelector("#manual-meta");
const manualMessage = document.querySelector("#manual-message");
const uploadLabel = document.querySelector("#upload-label");
const exampleButtons = document.querySelectorAll("[data-question]");

let manualReady = false;
let uploadingManual = false;

function setLoading(isLoading) {
  askButton.disabled = isLoading || !manualReady || uploadingManual;
  buttonLabel.textContent = isLoading ? "Searching the manual…" : "Ask Page Pilot";
}

function setManualReady(isReady) {
  manualReady = isReady;
  questionInput.disabled = !isReady || uploadingManual;
  askButton.disabled = !isReady || uploadingManual;
  exampleButtons.forEach((button) => {
    button.disabled = !isReady || uploadingManual;
  });
}

function renderManualStatus(status) {
  setManualReady(status.ready);
  if (status.ready) {
    manualName.textContent = status.filename;
    manualMeta.textContent = `${status.page_count} pages · ${status.chunk_count} searchable chunks`;
    uploadLabel.textContent = "Replace PDF";
  } else {
    manualName.textContent = "No manual uploaded";
    manualMeta.textContent = "Upload a text-based PDF to start asking questions.";
    uploadLabel.textContent = "Upload PDF";
  }
}

function showManualMessage(text, isError = false) {
  manualMessage.textContent = text;
  manualMessage.classList.toggle("error", isError);
  manualMessage.hidden = false;
}

async function loadManualStatus() {
  try {
    const response = await fetch("/manual");
    if (!response.ok) throw new Error();
    renderManualStatus(await response.json());
  } catch {
    renderManualStatus({ ready: false });
    showManualMessage("Could not read the active manual status.", true);
  }
}

async function uploadManual(file) {
  uploadingManual = true;
  setManualReady(manualReady);
  uploadLabel.textContent = "Indexing…";
  showManualMessage("Extracting pages and creating embeddings. This can take a moment.");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/manual", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "The PDF could not be processed.");
    }

    resultCard.hidden = true;
    message.hidden = true;
    renderManualStatus(data);
    showManualMessage(`${data.filename} is indexed and ready for questions.`);
  } catch (error) {
    showManualMessage(error.message || "The PDF could not be processed.", true);
    await loadManualStatus();
  } finally {
    uploadingManual = false;
    manualFile.value = "";
    setManualReady(manualReady);
    uploadLabel.textContent = manualReady ? "Replace PDF" : "Upload PDF";
  }
}

function showError(text) {
  message.textContent = text;
  message.hidden = false;
  resultCard.hidden = true;
}

function renderPages(pages) {
  sourcePages.replaceChildren();

  if (!pages.length) {
    const empty = document.createElement("span");
    empty.className = "no-sources";
    empty.textContent = "No supporting page found";
    sourcePages.append(empty);
    return;
  }

  pages.forEach((page) => {
    const chip = document.createElement("span");
    chip.className = "page-chip";
    chip.textContent = `Page ${page}`;
    sourcePages.append(chip);
  });
}

function renderResult(data) {
  message.hidden = true;
  answerStatus.textContent = data.answerable ? "Found in manual" : "Not found in manual";
  answerStatus.classList.toggle("not-found", !data.answerable);
  answerText.textContent = data.answer;
  retrievedPages.textContent = data.retrieved_pages.join(", ") || "none";
  renderPages(data.source_pages);
  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function askQuestion(question) {
  setLoading(true);
  message.hidden = true;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 2 }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Page Pilot could not answer the question.");
    }

    renderResult(data);
  } catch (error) {
    showError(error.message || "Could not connect to Page Pilot.");
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) askQuestion(question);
});

questionInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    form.requestSubmit();
  }
});

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
});

manualFile.addEventListener("change", () => {
  const [file] = manualFile.files;
  if (file) uploadManual(file);
});

fetch("/health")
  .then((response) => {
    if (!response.ok) throw new Error();
    serverStatus.classList.add("online");
    serverStatus.lastElementChild.textContent = "API online";
  })
  .catch(() => {
    serverStatus.classList.add("offline");
    serverStatus.lastElementChild.textContent = "API offline";
  });

loadManualStatus();
