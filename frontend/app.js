const taskForm = document.getElementById("task-form");
const taskInput = document.getElementById("task-input");
const runButton = document.getElementById("run-button");
const classificationEl = document.getElementById("classification");
const selectedModelEl = document.getElementById("selected-model");
const statusEl = document.getElementById("run-status");
const tokensUsedEl = document.getElementById("tokens-used");
const tokensRemainingEl = document.getElementById("tokens-remaining");
const meterFill = document.getElementById("meter-fill");
const smallBudgetEl = document.getElementById("small-budget");
const largeBudgetEl = document.getElementById("large-budget");
const estimatedCostEl = document.getElementById("estimated-cost");
const largeTokensAvoidedEl = document.getElementById("large-tokens-avoided");
const taskBudgetSavedEl = document.getElementById("task-budget-saved");
const usdSavedEl = document.getElementById("usd-saved");
const routeLog = document.getElementById("route-log");
const stepLog = document.getElementById("step-log");
const subtaskLog = document.getElementById("subtask-log");
const agentLog = document.getElementById("agent-log");
const finalOutput = document.getElementById("final-output");

let source = null;
let budgetTotal = 0;

function appendLog(listEl, text) {
  const li = document.createElement("li");
  li.textContent = text;
  listEl.prepend(li);
}

function updateBudget(used, remaining, possibleTotal) {
  if (possibleTotal > 0) {
    budgetTotal = possibleTotal;
  }
  tokensUsedEl.textContent = String(used ?? 0);
  tokensRemainingEl.textContent = String(remaining ?? 0);

  const total = budgetTotal || (Number(used || 0) + Number(remaining || 0));
  const ratio = total > 0 ? Math.min(100, Math.max(0, (Number(used || 0) / total) * 100)) : 0;
  meterFill.style.width = `${ratio.toFixed(1)}%`;
}

function updateModelMetrics(modelBudgets, savings) {
  if (modelBudgets?.small) {
    const small = modelBudgets.small;
    smallBudgetEl.textContent = `used ${small.used_tokens ?? 0} / cap ${small.cap_tokens ?? 0}`;
  }
  if (modelBudgets?.large) {
    const large = modelBudgets.large;
    largeBudgetEl.textContent = `used ${large.used_tokens ?? 0} / cap ${large.cap_tokens ?? 0}`;
    const smallCost = Number(modelBudgets?.small?.estimated_cost_usd ?? 0);
    const largeCost = Number(large.estimated_cost_usd ?? 0);
    estimatedCostEl.textContent = `$${(smallCost + largeCost).toFixed(6)}`;
  }
  if (savings) {
    largeTokensAvoidedEl.textContent = String(savings.large_tokens_avoided ?? 0);
    taskBudgetSavedEl.textContent = String(savings.task_budget_saved_tokens ?? 0);
    usdSavedEl.textContent = `$${Number(savings.estimated_usd_saved_vs_large_only ?? 0).toFixed(6)}`;
  }
}

function applyEvent(payload, type) {
  if (payload.classification) {
    classificationEl.textContent = payload.classification;
  }
  if (payload.model) {
    selectedModelEl.textContent = payload.model;
  }
  if (typeof payload.tokens_used === "number" || typeof payload.tokens_remaining === "number") {
    const total = payload.step_data?.budget_total;
    updateBudget(payload.tokens_used, payload.tokens_remaining, total);
  }

  if (type === "route_decision") {
    appendLog(routeLog, payload.message || "Route updated");
  }
  if (type === "step") {
    const node = payload.step_data?.node ? `[${payload.step_data.node}] ` : "";
    appendLog(stepLog, `${node}${payload.message || "Step completed"}`);
    if (payload.step_data?.subtask) {
      const sub = payload.step_data.subtask;
      appendLog(
        subtaskLog,
        `#${sub.id} [${sub.classification}] via ${sub.model}: ${sub.text || "(subtask)"}`
      );
    }
  }
  if (type === "state_update" && payload.step_data?.status) {
    statusEl.textContent = payload.step_data.status;
  }
  if (type === "token_update") {
    const maybeTotal = payload.step_data?.budget_total;
    updateBudget(payload.step_data?.tokens_used ?? payload.tokens_used, payload.step_data?.tokens_remaining ?? payload.tokens_remaining, maybeTotal);
  }
  if (payload.step_data?.model_budgets || payload.step_data?.orchestration_savings) {
    updateModelMetrics(payload.step_data?.model_budgets, payload.step_data?.orchestration_savings);
  }
  if (payload.step_data?.subtask_results) {
    subtaskLog.innerHTML = "";
    payload.step_data.subtask_results.forEach((sub) => {
      appendLog(
        subtaskLog,
        `#${sub.id} [${sub.classification}] via ${sub.model}: ${sub.text || "(subtask)"}`
      );
    });
  }
  if (payload.step_data?.agent_messages) {
    agentLog.innerHTML = "";
    payload.step_data.agent_messages.slice(-12).forEach((msg) => appendLog(agentLog, msg));
  }
  if (type === "error") {
    statusEl.textContent = "error";
    appendLog(stepLog, payload.message || "Run failed");
  }
}

function closeSource() {
  if (source) {
    source.close();
    source = null;
  }
}

function openStream(runId) {
  closeSource();
  source = new EventSource(`/api/stream/${runId}`);

  const eventTypes = ["state_update", "step", "route_decision", "token_update", "error", "done"];
  eventTypes.forEach((type) => {
    source.addEventListener(type, (event) => {
      const payload = JSON.parse(event.data);
      applyEvent(payload, type);

      if (type === "done") {
        statusEl.textContent = payload.step_data?.status || "completed";
        finalOutput.textContent = payload.step_data?.final_output || "(empty)";
        updateModelMetrics(payload.step_data?.model_budgets, payload.step_data?.orchestration_savings);
        if (payload.step_data?.subtask_results) {
          subtaskLog.innerHTML = "";
          payload.step_data.subtask_results.forEach((sub) => {
            appendLog(
              subtaskLog,
              `#${sub.id} [${sub.classification}] via ${sub.model}: ${sub.text || "(subtask)"}`
            );
          });
        }
        if (payload.step_data?.agent_messages) {
          agentLog.innerHTML = "";
          payload.step_data.agent_messages.slice(-12).forEach((msg) => appendLog(agentLog, msg));
        }
        closeSource();
        runButton.disabled = false;
      }
      if (type === "error") {
        closeSource();
        runButton.disabled = false;
      }
    });
  });

  source.onerror = () => {
    appendLog(stepLog, "Stream connection interrupted.");
  };
}

function resetUi() {
  classificationEl.textContent = "-";
  selectedModelEl.textContent = "-";
  statusEl.textContent = "running";
  tokensUsedEl.textContent = "0";
  tokensRemainingEl.textContent = "0";
  meterFill.style.width = "0%";
  smallBudgetEl.textContent = "used 0 / cap 0";
  largeBudgetEl.textContent = "used 0 / cap 0";
  estimatedCostEl.textContent = "$0.000000";
  largeTokensAvoidedEl.textContent = "0";
  taskBudgetSavedEl.textContent = "0";
  usdSavedEl.textContent = "$0.000000";
  routeLog.innerHTML = "";
  stepLog.innerHTML = "";
  subtaskLog.innerHTML = "";
  agentLog.innerHTML = "";
  finalOutput.textContent = "Running...";
  budgetTotal = 0;
}

taskForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const task = taskInput.value.trim();
  if (!task) {
    return;
  }

  resetUi();
  runButton.disabled = true;

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
    });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const payload = await response.json();
    openStream(payload.run_id);
  } catch (error) {
    statusEl.textContent = "error";
    finalOutput.textContent = `Failed to start run: ${error.message}`;
    runButton.disabled = false;
  }
});
