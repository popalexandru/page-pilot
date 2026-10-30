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

function setLoading(isLoading) {
  askButton.disabled = isLoading;
  buttonLabel.textContent = isLoading ? "Searching the manual…" : "Ask Page Pilot";
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

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question;
    questionInput.focus();
  });
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
