"use strict";

const $ = (sel) => document.querySelector(sel);

let toastTimer = null;
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 6000);
}

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.detail ?? `${res.status} ${res.statusText}`);
  return data;
}

/** Run an async action with the button disabled, surfacing errors as a toast. */
async function withButton(button, fn) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Working…";
  try {
    return await fn();
  } catch (err) {
    toast(err.message);
    return null;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

// ------------------------------------------------------------------ tabs

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".panel").forEach((p) =>
      p.classList.toggle("active", p.id === tab.dataset.panel));
  });
});

// ---------------------------------------------------------------- status

const POSE_ROWS = [
  ["roll", "roll", "°"], ["pitch", "pitch", "°"], ["yaw", "yaw", "°"],
  ["x", "x_mm", "mm"], ["y", "y_mm", "mm"], ["z", "z_mm", "mm"],
];

function fillTable(table, rows) {
  const body = table.querySelector("tbody");
  body.textContent = "";
  for (const cells of rows) {
    const tr = document.createElement("tr");
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.append(td);
    }
    body.append(tr);
  }
}

async function refreshStatus() {
  const conn = $("#conn");
  try {
    const s = await api("/api/status");
    conn.textContent = s.busy ? "busy" : "connected";
    conn.className = `pill ${s.busy ? "pill-idle" : "pill-ok"}`;

    fillTable($("#pose-table"), [
      ...POSE_ROWS.map(([label, key, unit]) => [label, `${s.head_pose[key].toFixed(2)} ${unit}`]),
      ["antenna L", `${s.antennas_deg[0].toFixed(2)} °`],
      ["antenna R", `${s.antennas_deg[1].toFixed(2)} °`],
    ]);

    fillTable($("#imu-table"), s.imu
      ? [
          ["accel", s.imu.accelerometer.map((v) => v.toFixed(2)).join(", ")],
          ["gyro", s.imu.gyroscope.map((v) => v.toFixed(3)).join(", ")],
          ["quat", s.imu.quaternion.map((v) => v.toFixed(3)).join(", ")],
          ["temp", `${Number(s.imu.temperature).toFixed(1)} °C`],
        ]
      : [["", "no IMU on this unit"]]);
  } catch (err) {
    conn.textContent = "disconnected";
    conn.className = "pill pill-bad";
  }
}

async function refreshMotors() {
  try {
    const m = await api("/api/motor_status");
    $("#motor-count").textContent = `(${m.count})`;
    fillTable($("#motor-table"), m.motors.map((mo) =>
      [mo.name, mo.position_deg.toFixed(2), mo.position_rad.toFixed(4)]));
  } catch { /* the status pill already reports the outage */ }
}

// -------------------------------------------------------------- movement

const HEAD_AXES = [
  { key: "roll", min: -40, max: 40, unit: "°" },
  { key: "pitch", min: -40, max: 40, unit: "°" },
  { key: "yaw", min: -40, max: 40, unit: "°" },
  { key: "body_yaw", min: -40, max: 40, unit: "°" },
  { key: "x", min: -25, max: 25, unit: "mm" },
  { key: "y", min: -25, max: 25, unit: "mm" },
  { key: "z", min: -25, max: 25, unit: "mm" },
  { key: "duration", min: 0.1, max: 5, unit: "s", value: 0.5, step: 0.1 },
];

const ANTENNA_AXES = [
  { key: "left", min: -150, max: 150, unit: "°" },
  { key: "right", min: -150, max: 150, unit: "°" },
  { key: "duration", min: 0.1, max: 5, unit: "s", value: 0.5, step: 0.1 },
];

/** Build a labelled slider row and return a getter for its current value. */
function buildSliders(container, axes) {
  container.textContent = "";
  const inputs = {};
  for (const axis of axes) {
    const value = axis.value ?? 0;
    const input = Object.assign(document.createElement("input"), {
      type: "range", min: axis.min, max: axis.max,
      step: axis.step ?? 1, value,
    });
    const out = document.createElement("output");
    out.textContent = `${value} ${axis.unit}`;
    input.addEventListener("input", () => { out.textContent = `${input.value} ${axis.unit}`; });

    const row = document.createElement("div");
    row.className = "slider";
    const name = document.createElement("span");
    name.textContent = axis.key;
    row.append(name, input, out);
    container.append(row);

    inputs[axis.key] = { input, out, reset: () => {
      input.value = value;
      out.textContent = `${value} ${axis.unit}`;
    } };
  }
  return {
    values: () => Object.fromEntries(
      Object.entries(inputs).map(([k, v]) => [k, Number(v.input.value)])),
    reset: () => Object.values(inputs).forEach((v) => v.reset()),
  };
}

const headSliders = buildSliders($("#head-sliders"), HEAD_AXES);
const antennaSliders = buildSliders($("#antenna-sliders"), ANTENNA_AXES);

$("#head-apply").addEventListener("click", (e) =>
  withButton(e.target, () => api("/api/move_head", { method: "POST", body: headSliders.values() })));
$("#head-reset").addEventListener("click", () => headSliders.reset());
$("#antenna-apply").addEventListener("click", (e) =>
  withButton(e.target, () => api("/api/move_antennas", { method: "POST", body: antennaSliders.values() })));

document.querySelectorAll("[data-post]").forEach((button) => {
  button.addEventListener("click", () =>
    withButton(button, () => api(button.dataset.post, { method: "POST" })));
});

// ---------------------------------------------------------------- camera

const streamImg = $("#stream");
const streamToggle = $("#stream-toggle");

streamToggle.addEventListener("click", () => {
  const on = streamImg.hidden;
  // Cache-bust so a restarted stream is not served from the browser cache.
  streamImg.src = on ? `/api/camera/stream?t=${Date.now()}` : "";
  streamImg.hidden = !on;
  streamToggle.textContent = on ? "Stop stream" : "Start stream";
  $("#stream-hint").textContent = on ? "" : "Stream is off.";
});

$("#capture-save").addEventListener("click", (e) =>
  withButton(e.target, async () => {
    const r = await api("/api/camera/save", { method: "POST" });
    toast(`Saved ${r.filename}`);
    await refreshCaptures();
  }));

/** Render a file list with download and delete controls. */
function renderFiles(list, files, { download, remove, play }) {
  list.textContent = "";
  if (!files.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "Nothing here yet.";
    list.append(li);
    return;
  }
  for (const f of files) {
    const li = document.createElement("li");

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = f.filename;

    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${(f.size / 1024).toFixed(0)} KiB · ${f.modified.replace("T", " ")}`;

    const dl = document.createElement("a");
    dl.href = download(f.filename);
    dl.download = f.filename;
    dl.append(Object.assign(document.createElement("button"), { textContent: "Download" }));

    li.append(name, meta, dl);

    if (play) {
      const btn = Object.assign(document.createElement("button"), { textContent: "Play" });
      btn.addEventListener("click", () => withButton(btn, () => play(f.filename)));
      li.append(btn);
    }

    const del = Object.assign(document.createElement("button"), { textContent: "Delete" });
    del.addEventListener("click", () => withButton(del, async () => {
      await remove(f.filename);
      await refreshFiles();
    }));
    li.append(del);

    list.append(li);
  }
}

async function refreshCaptures() {
  const { captures } = await api("/api/camera/list");
  renderFiles($("#capture-list"), captures, {
    download: (n) => `/api/camera/download/${encodeURIComponent(n)}`,
    remove: (n) => api(`/api/camera/delete/${encodeURIComponent(n)}`, { method: "DELETE" }),
  });
}

// ----------------------------------------------------------------- audio

const recordToggle = $("#record-toggle");
const recordTimer = $("#record-timer");
let recordingSince = null;
let recordingTicker = null;

recordToggle.addEventListener("click", async () => {
  // withButton restores the original label, so set the next one afterwards.
  await withButton(recordToggle, async () => {
    if (recordingSince === null) {
      await api("/api/audio/start_recording", { method: "POST" });
      recordingSince = Date.now();
      recordingTicker = setInterval(() => {
        recordTimer.textContent = `${((Date.now() - recordingSince) / 1000).toFixed(1)} s`;
      }, 100);
    } else {
      const r = await api("/api/audio/stop_recording", { method: "POST" });
      clearInterval(recordingTicker);
      recordingSince = null;
      recordTimer.textContent = "";
      toast(`Saved ${r.filename} — ${r.seconds.toFixed(1)} s, peak ${r.peak.toFixed(4)}`);
      await refreshRecordings();
    }
  });
  recordToggle.textContent = recordingSince === null ? "Start recording" : "Stop recording";
});

async function refreshRecordings() {
  const { recordings } = await api("/api/audio/list");
  renderFiles($("#recording-list"), recordings, {
    download: (n) => `/api/audio/download/${encodeURIComponent(n)}`,
    remove: (n) => api(`/api/audio/delete/${encodeURIComponent(n)}`, { method: "DELETE" }),
    play: (n) => api(`/api/audio/play/${encodeURIComponent(n)}`, { method: "POST" }),
  });
}

async function refreshFiles() {
  await Promise.all([refreshCaptures(), refreshRecordings()]);
}

// ----------------------------------------------------------------- tests

/** Show a PASS/FAIL verdict plus the raw payload for anything the UI omits. */
function showResult(target, { passed, summary, detail, raw }) {
  target.textContent = "";
  const verdict = document.createElement("span");
  verdict.className = `verdict ${passed ? "pass" : "fail"}`;
  verdict.textContent = passed ? "PASS" : "FAIL";

  const line = document.createElement("div");
  line.append(verdict, document.createTextNode(` ${summary}`));
  target.append(line);

  if (detail) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = detail;
    target.append(p);
  }

  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(raw, null, 2);
  target.append(pre);
}

$("#cal-run").addEventListener("click", (e) => withButton(e.target, async () => {
  const r = await api("/api/test/calibration", {
    method: "POST",
    body: {
      amplitude: Number($("#cal-amplitude").value),
      steps: Number($("#cal-steps").value),
      antennas: $("#cal-antennas").checked,
    },
  });
  const failed = r.checks.filter((c) => !c.passed).map((c) => c.name);
  showResult($("#cal-result"), {
    passed: r.passed,
    summary: `${r.checks.length} checks`,
    detail: failed.length ? `failed: ${failed.join(", ")}` : "all checks within tolerance",
    raw: r,
  });
}));

$("#rot-run").addEventListener("click", (e) => withButton(e.target, async () => {
  const r = await api("/api/test/rotation_validation", {
    method: "POST",
    body: {
      axis: $("#rot-axis").value,
      angle: Number($("#rot-angle").value),
      tolerance: Number($("#rot-tol").value),
    },
  });
  showResult($("#rot-result"), {
    passed: Boolean(r.passed),
    summary: r.ok ? `${r.measured_deg.toFixed(2)}° measured vs ${r.expected_deg.toFixed(1)}° commanded` : "measurement failed",
    detail: r.detail,
    raw: r,
  });
}));

$("#vs-run").addEventListener("click", (e) => withButton(e.target, async () => {
  const r = await api("/api/test/visual_scale", {
    method: "POST",
    body: { axis: $("#vs-axis").value },
  });
  showResult($("#vs-result"), {
    passed: Boolean(r.ok),
    summary: r.ok ? `${r.px_per_deg.toFixed(2)} px/deg (effective fx ${r.fx_effective.toFixed(1)})` : "fit failed",
    detail: r.ok ? `rms residual ${r.rms_residual_px.toFixed(2)} px over ${r.samples.length} angles` : r.error,
    raw: r,
  });
}));

// ------------------------------------------------------------------ boot

async function loadLastResults() {
  const [rot, cal] = await Promise.all([
    api("/api/test/last_rotation_result").catch(() => null),
    api("/api/test/last_calibration_result").catch(() => null),
  ]);
  if (rot?.result) {
    showResult($("#rot-result"), {
      passed: Boolean(rot.result.passed),
      summary: "last run",
      detail: rot.result.detail,
      raw: rot.result,
    });
  }
  if (cal?.result) {
    showResult($("#cal-result"), {
      passed: cal.result.passed,
      summary: "last run",
      detail: "",
      raw: cal.result,
    });
  }
}

refreshStatus();
refreshMotors();
refreshFiles().catch((err) => toast(err.message));
loadLastResults();
setInterval(refreshStatus, 500);
setInterval(refreshMotors, 1000);
