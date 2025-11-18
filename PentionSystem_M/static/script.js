let map;
let vanMarker = null;
let sourceCircle = null;
let vanPath = null;
let pathLatLngs = [];
let ws = null;

const btnStart = document.getElementById("btn-start");
const btnReset = document.getElementById("btn-reset");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const simIdEl = document.getElementById("sim-id");
const modelVersionEl = document.getElementById("model-version");
const driftScoreEl = document.getElementById("drift-score");
const latencyEl = document.getElementById("latency-ms");
const bundleNameEl = document.getElementById("bundle-name");
const bundleLogEl = document.getElementById("bundle-log");
const processingCard = document.getElementById("processing-card");
const simCard = document.getElementById("sim-card");
const bundleCard = document.getElementById("bundle-card");
const stabilityEl = document.getElementById("stability-index");
const confidenceEl = document.getElementById("confidence");


function showSimCard() { simCard.style.display = "block"; }
function hideSimCard() { simCard.style.display = "none"; }

function showBundleCard() { bundleCard.style.display = "block"; }
function hideBundleCard() { bundleCard.style.display = "none"; }

function showProcessing() {
  processingCard.style.display = "block";
}

function hideProcessing() {
  processingCard.style.display = "none";
}

document.getElementById("btn-debug").addEventListener("click", async () => {
  showLoading();
  btnStart.disabled = true;
  btnReset.disabled = true;

  try {
    const resp = await fetch("/api/start_simulation_near", { method: "POST" });
    if (!resp.ok) {
      hideLoading();
      btnStart.disabled = false;
      btnReset.disabled = false;
      alert("Error starting debug simulation: " + resp.status);
    }
  } catch (e) {
    hideLoading();
    btnStart.disabled = false;
    btnReset.disabled = false;
    alert("Error starting debug simulation: " + e);
  }
});

function showLoading() {
  document.getElementById("loading-screen").style.visibility = "visible";
}

function hideLoading() {
  document.getElementById("loading-screen").style.visibility = "hidden";
}

function setStatus(state) {
  statusPill.classList.remove("status-idle", "status-patrolling", "status-detected");
  if (state === "patrolling") {
    statusPill.classList.add("status-patrolling");
    statusText.textContent = "Patrolling (van in movimento)";
  } else if (state === "detected") {
    statusPill.classList.add("status-detected");
    statusText.textContent = "Detection – pipeline in esecuzione";
  } else {
    statusPill.classList.add("status-idle");
    statusText.textContent = "Idle";
  }
}

function ensureMap() {
  if (!map) {
    map = L.map("map").setView([52.372, 4.900], 13);

    const tileLayer = L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
      }
    );
    tileLayer.addTo(map);
    hideLoading();
  }
}

function resetGraphics() {
  if (vanMarker) {
    map.removeLayer(vanMarker);
    vanMarker = null;
  }
  if (sourceCircle) {
    map.removeLayer(sourceCircle);
    sourceCircle = null;
  }
  if (vanPath) {
    map.removeLayer(vanPath);
    vanPath = null;
  }
  pathLatLngs = [];
  simIdEl.textContent = "–";
  //simDistEl.textContent = "–";
  modelVersionEl.textContent = "–";
  driftScoreEl.textContent = "–";
  latencyEl.textContent = "–";
  stabilityEl.textContent = "–";
  confidenceEl.textContent = "–";
}


function renderBundleSummary(bundle) {
  if (!bundle || !bundle.event) return "No bundle data.";

  const ev = bundle.event;

  return `
    <div class="metric-row">
      <div class="metric-label">Timestamp</div>
      <div class="metric-value">${ev.timestamp}</div>
    </div>

    <div class="metric-row">
      <div class="metric-label">Substance</div>
      <div class="metric-value">${ev.SensorSubstance.compound_name}</div>
    </div>

    <div class="metric-row">
      <div class="metric-label">Confidence</div>
      <div class="metric-value">${ev.Inference.confidence_score}</div>
    </div>

    <div class="metric-row">
      <div class="metric-label">Wind</div>
      <div class="metric-value">${ev.SensorAir.wind_speed_mps} m/s @ ${ev.SensorAir.wind_dir_deg}°</div>
    </div>

    <div class="metric-row">
      <div class="metric-label">GPS</div>
      <div class="metric-value">${Number(ev.SensorGPS.latitude).toFixed(5)}, ${Number(ev.SensorGPS.longitude).toFixed(5)}</div>
    </div>

    <div class="metric-row">
      <div class="metric-label">Hash</div>
      <div class="metric-value">${bundle.hash_sha256.slice(0,12)}...</div>
    </div>

    <button class="small-btn" onclick="openReportPopup()">View full report</button>
  `;
}


function openReportPopup() {
  const bundle = window.lastBundle;
  const monitoring = window.lastMonitoring;
  if (!bundle || !monitoring) return;

  const ev = bundle.event;

  const content = document.getElementById("popup-content");

  content.innerHTML = `
    <div class="report-section">
      <h3>General Info</h3>
      <div class="report-grid">
        <div>
          <div class="report-item-label">Simulation ID</div>
          <div class="report-item-value">${ev.simulation_id}</div>
        </div>
        <div>
          <div class="report-item-label">Timestamp</div>
          <div class="report-item-value">${ev.timestamp}</div>
        </div>
      </div>
    </div>

    <div class="report-section">
      <h3>Detected Substance</h3>
      <div class="report-grid">
        <div>
          <div class="report-item-label">Substance</div>
          <div class="report-item-value">${ev.SensorSubstance.compound_name}</div>
        </div>
        <div>
          <div class="report-item-label">Confidence</div>
          <div class="report-item-value">${ev.Inference.confidence_score}</div>
        </div>
      </div>
    </div>

    <div class="report-section">
      <h3>Environmental Conditions</h3>
      <div class="report-grid">
        <div>
          <div class="report-item-label">Temperature</div>
          <div class="report-item-value">${ev.SensorAir.temperature_C} °C</div>
        </div>
        <div>
          <div class="report-item-label">Humidity</div>
          <div class="report-item-value">${ev.SensorAir["humidity_%"]}</div>
        </div>
        <div>
          <div class="report-item-label">Wind</div>
          <div class="report-item-value">${ev.SensorAir.wind_speed_mps} m/s @ ${ev.SensorAir.wind_dir_deg}°</div>
        </div>
        <div>
          <div class="report-item-label">Stability Class</div>
          <div class="report-item-value">${ev.SensorAir.stability_class}</div>
        </div>
      </div>
    </div>

    <div class="report-section">
      <h3>GPS</h3>
      <div class="report-grid">
        <div>
          <div class="report-item-label">Latitude</div>
          <div class="report-item-value">${ev.SensorGPS.latitude}</div>
        </div>
        <div>
          <div class="report-item-label">Longitude</div>
          <div class="report-item-value">${ev.SensorGPS.longitude}</div>
        </div>
      </div>
    </div>

    <div class="report-section">
      <h3>Monitoring</h3>
      <div class="report-grid">
        <div>
          <div class="report-item-label">Model version</div>
          <div class="report-item-value">${monitoring.model_version}</div>
        </div>
        <div>
          <div class="report-item-label">Latency</div>
          <div class="report-item-value">${monitoring.latency_ms} ms</div>
        </div>
        <div>
          <div class="report-item-label">Drift score</div>
          <div class="report-item-value">${monitoring.drift_score}</div>
        </div>
        <div>
          <div class="report-item-label">Stability index</div>
          <div class="report-item-value">${monitoring.stability_index}</div>
        </div>
      </div>
    </div>
  `;

  document.getElementById("report-popup").style.display = "flex";
}


function closeReportPopup() {
  document.getElementById("report-popup").style.display = "none";
}

function downloadBundleJson() {
  const bundle = window.lastBundle;
  if (!bundle) return;

  const ev = bundle.event || {};
  const simId = ev.simulation_id || "simulation";

  const blob = new Blob([JSON.stringify(bundle, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `forensic_bundle_${simId}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  URL.revokeObjectURL(url);
}

function formatHash(hash) {
    if (!hash || typeof hash !== "string") return String(hash);

    // Rimuove eventuali spazi o prefissi strani
    const clean = hash.replace(/\s+/g, "");

    // Divide in gruppi da 4 caratteri
    const grouped = clean.match(/.{1,4}/g) || [];

    // Ritorna stringa a gruppi separati da spazio
    return grouped.join(" ");
}

async function exportPDF() {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: "pt", format: "a4" });

  const bundle = window.lastBundle;
  const monitoring = window.lastMonitoring;
  if (!bundle || !monitoring) return;

  const ev = bundle.event;

  let y = 40;

  // LOGO
  try {
      const img = await fetch("/static/logo.png")
        .then(r => r.blob())
        .then(b => new Promise(resolve => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(b);
        }));

      const imgWidth = 180;     // larghezza logo
      const imgHeight = 80;     // altezza logo
      const pageWidth = doc.internal.pageSize.getWidth();

      // centro orizzontale
      const x = (pageWidth - imgWidth) / 2;

      doc.addImage(img, "PNG", x, y, imgWidth, imgHeight);
      y += imgHeight + 30; // spazio sotto il logo

      doc.setFontSize(26);
      doc.text("Forensic Detection Report", pageWidth / 2, y, { align: "center" });
      y += 40;
  } catch (e) {}
  const checkPage = (extra = 30) => {
    const pageHeight = doc.internal.pageSize.getHeight();
    if (y + extra >= pageHeight - 40) {
      doc.addPage();
      y = 40;
    }
  };
  const addSection = (title) => {
    checkPage(60);

    y += 16;
    doc.setFontSize(16);
    doc.text(title, 40, y);

    y += 6;
    doc.setLineWidth(0.5);
    doc.line(40, y, 550, y);

    y += 16;
  };

  const addField = (label, value, opts = {}) => {
    const { isHash = false } = opts;

    const pageWidth = doc.internal.pageSize.getWidth();
    const marginX = 50;
    const maxWidth = pageWidth - marginX - 40;

    checkPage(50);  // controllo prima di scrivere il blocco

    // LABEL
    doc.setFontSize(11);
    doc.text(label + ":", marginX, y);
    y += 14;

    // HASH (piccolo + formattato)
    if (isHash) {
      value = formatHash(value);
      doc.setFontSize(9);
    } else {
      doc.setFontSize(11);
    }

    // WRAPPING MULTILINE
    const lines = doc.splitTextToSize(String(value), maxWidth);

    for (let line of lines) {
      checkPage(20);
      doc.text(line, marginX + 20, y);
      y += 14;
    }

    y += 6;
  };

  // --- Contenuto report ---

  addSection("General Info");
  addField("Simulation ID", ev.simulation_id);
  addField("Timestamp", ev.timestamp);

  addSection("Detected Substance");
  addField("Name", ev.SensorSubstance.compound_name);
  addField("Confidence", ev.Inference.confidence_score);

  addSection("Environmental Conditions");
  addField("Temperature (°C)", ev.SensorAir.temperature_C);
  addField("Humidity (%)", ev.SensorAir["humidity_%"]);
  addField("Wind", `${ev.SensorAir.wind_speed_mps} m/s @ ${ev.SensorAir.wind_dir_deg}°`);
  addField("Stability Class", ev.SensorAir.stability_class);

  addSection("GPS");
  addField("Latitude", ev.SensorGPS.latitude);
  addField("Longitude", ev.SensorGPS.longitude);

  addSection("Monitoring");
  addField("Model Version", monitoring.model_version);
  addField("Latency (ms)", monitoring.latency_ms);
  addField("Drift Score", monitoring.drift_score);
  addField("Stability Index", monitoring.stability_index);

  addSection("Security & Audit");
  addField("Forensic bundle hash (SHA256)", bundle.hash_sha256, { isHash: true });
  addField("Bundle name", bundle.bundle_name);
  addField("Bundle signature", bundle.signature);

  doc.save(`forensic_report_${ev.simulation_id}.pdf`);
}

function colorizeMetric(el, value, thresholds) {
  // pulisci eventuali classi precedenti
  el.classList.remove("badge-green", "badge-yellow", "badge-red", "updated");

  if (value >= thresholds.red) {
    el.classList.add("badge-red");
  } else if (value >= thresholds.yellow) {
    el.classList.add("badge-yellow");
  } else {
    el.classList.add("badge-green");
  }

  // piccolo effetto "dash"
  el.classList.add("updated");
  setTimeout(() => el.classList.remove("updated"), 300);
}

function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    return;
  }

  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${window.location.host}/ws`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "init") {
      simIdEl.textContent = msg.simulation_id || "–";
      setStatus("patrolling");
      btnStart.disabled = true;
      btnReset.disabled = false;

      resetGraphics();
      const sLat = msg.source.lat;
      const sLon = msg.source.lon;
      const vLat = msg.van.lat;
      const vLon = msg.van.lon;

      sourceCircle = L.circle([sLat, sLon], {
        radius: 250,
        color: "#f97316",
        fillColor: "#fb923c",
        fillOpacity: 0.25,
      }).addTo(map);

      vanMarker = L.circleMarker([vLat, vLon], {
        radius: 6,
        color: "#38bdf8",
        fillColor: "#0ea5e9",
        fillOpacity: 1.0,
      }).addTo(map);

      pathLatLngs = [[vLat, vLon]];
      vanPath = L.polyline(pathLatLngs, {
        weight: 3,
        opacity: 0.9,
      }).addTo(map);

      map.fitBounds([[sLat, sLon], [vLat, vLon]], { padding: [40, 40] });

      hideLoading();
    }

    if (msg.type === "van_update") {
      if (!map || !vanMarker) return;

      const lat = msg.lat;
      const lon = msg.lon;

      vanMarker.setLatLng([lat, lon]);
      pathLatLngs.push([lat, lon]);
      vanPath.setLatLngs(pathLatLngs);

      // if (msg.distance_m !== undefined) {
      //   simDistEl.textContent = `${msg.distance_m.toFixed(1)} m`;
      // }

      if (msg.status === "detected") {
          setStatus("detected");
          showProcessing();
          btnStart.disabled = true;
          btnReset.disabled = true;
      } else if (msg.status === "patrolling") {
          setStatus("patrolling");
      }
    }

    if (msg.type === "detection_result") {
      showSimCard();
      hideProcessing();
      setStatus("idle");
      btnStart.disabled = false;
      btnReset.disabled = false;
      window.lastBundle = msg.forensic_bundle;
      window.lastMonitoring = msg.monitoring;

      if (msg.simulation_id) {
        simIdEl.textContent = msg.simulation_id;
      }

      if (msg.monitoring) {
        if (msg.monitoring.model_version) {
          modelVersionEl.textContent = msg.monitoring.model_version;
        }

        if (msg.monitoring.drift_score !== undefined) {
          const drift = msg.monitoring.drift_score;
          driftScoreEl.textContent = drift.toFixed ? drift.toFixed(3) : drift;
          // soglie esempio: >0.3 giallo, >0.6 rosso
          colorizeMetric(driftScoreEl, drift, { yellow: 0.3, red: 0.6 });
        }

        if (msg.monitoring.latency_ms !== undefined) {
          const latVal = msg.monitoring.latency_ms;
          latencyEl.textContent = latVal + " ms";
          colorizeMetric(latencyEl, latVal, { yellow: 80, red: 200 });
        }

        if (msg.monitoring.stability_index !== undefined) {
          const stab = msg.monitoring.stability_index;
          stabilityEl.textContent = stab.toFixed ? stab.toFixed(2) : stab;
          // opzionale: niente threshold, solo animazione leggera
          stabilityEl.classList.add("updated");
          setTimeout(() => stabilityEl.classList.remove("updated"), 300);
        }

        if (msg.monitoring.confidence !== undefined) {
          const c = msg.monitoring.confidence;
          confidenceEl.textContent = c.toFixed ? c.toFixed(2) : c;

          confidenceEl.classList.remove("badge-green", "badge-yellow", "badge-red", "updated");
          // qui high = good → logica invertita rispetto a drift/latency
          if (c >= 0.9) {
            confidenceEl.classList.add("badge-green");
          } else if (c >= 0.7) {
            confidenceEl.classList.add("badge-yellow");
          } else {
            confidenceEl.classList.add("badge-red");
          }
          confidenceEl.classList.add("updated");
          setTimeout(() => confidenceEl.classList.remove("updated"), 300);
        }
      }

      if (msg.registry && msg.registry.current_model_version) {
        modelVersionEl.textContent = msg.registry.current_model_version;
      }

      if (msg.forensic_bundle) {
        bundleNameEl.textContent = msg.forensic_bundle.bundle_name || "bundle";
        bundleLogEl.innerHTML = renderBundleSummary(msg.forensic_bundle);
        showBundleCard();
      }
    }


    if (msg.type === "sim_end") {
      setStatus("idle");
      btnStart.disabled = false;
      btnReset.disabled = false;
    }

    if (msg.type === "error") {
      setStatus("idle");
      btnStart.disabled = false;
      btnReset.disabled = false;
      hideLoading();
      alert("Simulation error: " + (msg.message || "unknown error"));
    }
  };

  ws.onclose = () => {
    setStatus("idle");
    btnStart.disabled = false;
    btnReset.disabled = false;
  };
}

btnStart.addEventListener("click", async () => {
  showLoading();
  connectWebSocket();
  await fetch("/api/start_simulation", { method: "POST" });
});

btnReset.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  setStatus("idle");
  btnStart.disabled = false;
  btnReset.disabled = true;
  resetGraphics();
  hideProcessing();
  hideLoading();
  hideSimCard();
  hideBundleCard();
});

ensureMap();
connectWebSocket();
hideLoading();
