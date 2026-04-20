// Backend API base URL — configurable via a data attribute, falls back to same origin
const API_BASE = window.__API_BASE__ || "http://localhost:8000";

// DOM references
const homeSelect = document.getElementById("home-team");
const awaySelect = document.getElementById("away-team");
const dateInput = document.getElementById("match-date");
const predictBtn = document.getElementById("predict-btn");
const resultSection = document.getElementById("result");
const errorSection = document.getElementById("error");

// Set default date to today
dateInput.valueAsDate = new Date();

// --- Load teams on page load ---
async function loadTeams() {
  try {
    const res = await fetch(`${API_BASE}/teams`);
    if (!res.ok) throw new Error(`Teams endpoint returned ${res.status}`);
    const data = await res.json();

    // Populate both dropdowns
    for (const select of [homeSelect, awaySelect]) {
      select.innerHTML = "";
      for (const team of data.teams) {
        const opt = document.createElement("option");
        opt.value = team.team_api_id;
        opt.textContent = team.name;
        select.appendChild(opt);
      }
      select.disabled = false;
    }

    // Pick sensible defaults (different teams so they can immediately predict)
    if (data.teams.length >= 2) {
      homeSelect.value = data.teams[0].team_api_id;
      awaySelect.value = data.teams[1].team_api_id;
    }

    predictBtn.disabled = false;
  } catch (err) {
    showError(`Failed to load teams: ${err.message}. Is the backend running at ${API_BASE}?`);
  }
}

// --- Predict on button click ---
async function predict() {
  const home_team_id = parseInt(homeSelect.value, 10);
  const away_team_id = parseInt(awaySelect.value, 10);

  if (home_team_id === away_team_id) {
    showError("Home and away teams must be different.");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting…";
  hideError();

  try {
    const body = {
      home_team_id,
      away_team_id,
      match_date: dateInput.value || null,
    };
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Status ${res.status}`);
    }
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(`Prediction failed: ${err.message}`);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Outcome";
  }
}

// --- Render the result section ---
function renderResult(data) {
  const headlineMap = {
    H: `${data.home_team} to win`,
    D: `Likely draw`,
    A: `${data.away_team} to win`,
  };
  document.getElementById("result-headline").textContent = headlineMap[data.prediction];
  document.getElementById("confidence-value").textContent = `${(data.confidence * 100).toFixed(1)}%`;

  // Probability bars
  for (const cls of ["H", "D", "A"]) {
    const pct = (data.probabilities[cls] * 100).toFixed(1);
    document.getElementById(`prob-${cls.toLowerCase()}`).style.width = `${pct}%`;
    document.getElementById(`prob-${cls.toLowerCase()}-value`).textContent = `${pct}%`;
  }

  // Meta
  document.getElementById("latency").textContent = `${data.inference_latency_ms} ms`;
  document.getElementById("model-version").textContent = data.model_version;
  document.getElementById("container-id").textContent = data.container_id;

  resultSection.classList.remove("hidden");
}

function showError(msg) {
  errorSection.textContent = msg;
  errorSection.classList.remove("hidden");
  resultSection.classList.add("hidden");
}

function hideError() {
  errorSection.classList.add("hidden");
}

// --- Wire up events ---
predictBtn.addEventListener("click", predict);

// Kick off on load
loadTeams();