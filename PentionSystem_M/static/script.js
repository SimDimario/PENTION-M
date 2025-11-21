let map;
let vanMarker = null;
let sourceCircle = null;
let vanPath = null;
let pathLatLngs = [];
let ws = null;
let vanIcon = null;

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
const btnDebug = document.getElementById("btn-debug");
const canvasRenderer = L.canvas({ padding: 0.5 });

function getJsPDF() {
  if (window.jspdf && window.jspdf.jsPDF) return window.jspdf.jsPDF;
  if (window.jsPDF) return window.jsPDF;
  return null;
}

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

  resetGraphics();
  hideSimCard();
  hideBundleCard();
  hideProcessing();
  window.lastBundle = null;
  window.lastMonitoring = null;


  btnStart.disabled = true;
  btnReset.disabled = true;
  btnDebug.disabled = true;

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

    // icona del laboratorio mobile
    vanIcon = L.icon({
      iconUrl: "/static/van_icon.png",
      iconSize: [32, 32],
      iconAnchor: [16, 16],
    });

    hideLoading();
  }
}

function resetGraphics() {
  window.lastBundle = null;
  window.lastMonitoring = null;

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
  const inf = ev.Inference || {};
  const sub = ev.SensorSubstance || {};

  const predicted = inf.predicted_class || "UNKNOWN";
  const trueName = sub.compound_name || "UNKNOWN";
  const confidence = inf.confidence_score || "N/A";

  return `
    <div class="metric-row">
      <div class="metric-label">True substance (sim)</div>
      <div class="metric-value">${trueName}</div>
    </div>

    <div class="metric-row">
      <div class="metric-label">Model prediction</div>
      <div class="metric-value">${predicted}</div>
    </div>

    <div class="metric-row">
      <div class="metric-label">Confidence</div>
      <div class="metric-value">${confidence}</div>
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
      <div class="metric-value">${bundle.hash_sha256.slice(0, 12)}...</div>
    </div>

    <button class="small-btn" onclick="openReportPopup()">View full report</button>
  `;
}

function openReportPopup() {
  const bundle = window.lastBundle;
  const monitoring = window.lastMonitoring;
  if (!bundle) return;

  const ev = bundle.event || {};

  const air = ev.SensorAir || {};
  const sub = ev.SensorSubstance || {};
  const gps = ev.SensorGPS || {};
  const piml = ev.PIML_Features || {};
  const inf = ev.Inference || {};
  const evMonitoring = ev.Monitoring || {};
  const modelOps = ev.ModelOps || {};
  const fexport = ev.ForensicExport || {};
  const artifacts = ev.artifacts || {};

  // monitoring esterno sempre prioritario
  const mon = monitoring || evMonitoring;

  const content = document.getElementById("popup-content");

  content.innerHTML = `
    <div class="report-section">
      <h3>Event metadata</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Simulation ID</div><div class="report-item-value">${ev.simulation_id}</div></div>
        <div><div class="report-item-label">Timestamp</div><div class="report-item-value">${ev.timestamp}</div></div>
        <div><div class="report-item-label">Bundle name</div><div class="report-item-value">${bundle.bundle_name}</div></div>
        <div><div class="report-item-label">Hash SHA256</div><div class="report-item-value">${bundle.hash_sha256}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Environmental conditions</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Temperature (°C)</div><div class="report-item-value">${air.temperature_C}</div></div>
        <div><div class="report-item-label">Humidity (%)</div><div class="report-item-value">${air["humidity_%"]}</div></div>
        <div><div class="report-item-label">Wind</div><div class="report-item-value">${air.wind_speed_mps} m/s @ ${air.wind_dir_deg}°</div></div>
        <div><div class="report-item-label">Stability class</div><div class="report-item-value">${air.stability_class}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Detected substance</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Compound</div><div class="report-item-value">${sub.compound_name}</div></div>
        <div><div class="report-item-label">Molecular formula</div><div class="report-item-value">${sub.molecular_formula}</div></div>
        <div><div class="report-item-label">Noise level</div><div class="report-item-value">${sub.noise_level}</div></div>
        <div><div class="report-item-label">EI Mass Spectrum (600 bins)</div><div class="report-item-value">${(sub.spectrum_ei_1_600||[]).join(", ")}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>PIML features</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Sigma_y</div><div class="report-item-value">${piml.sigma_y}</div></div>
        <div><div class="report-item-label">Sigma_z</div><div class="report-item-value">${piml.sigma_z}</div></div>
        <div><div class="report-item-label">Péclet number</div><div class="report-item-value">${piml.pe_number}</div></div>
        <div><div class="report-item-label">Stability index</div><div class="report-item-value">${piml.stability_index}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Location</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Latitude</div><div class="report-item-value">${gps.latitude}</div></div>
        <div><div class="report-item-label">Longitude</div><div class="report-item-value">${gps.longitude}</div></div>
        <div><div class="report-item-label">Altitude (m)</div><div class="report-item-value">${gps.altitude_m}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Inference</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Predicted class</div><div class="report-item-value">${inf.predicted_class}</div></div>
        <div><div class="report-item-label">Confidence</div><div class="report-item-value">${inf.confidence_score}</div></div>
        <div><div class="report-item-label">Dispersion map ID</div><div class="report-item-value">${inf.dispersion_map_id}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Monitoring</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Model version</div><div class="report-item-value">${mon.model_version}</div></div>
        <div><div class="report-item-label">Latency (ms)</div><div class="report-item-value">${mon.latency_ms}</div></div>
        <div><div class="report-item-label">Drift score</div><div class="report-item-value">${mon.drift_score}</div></div>
        <div><div class="report-item-label">MSE free</div><div class="report-item-value">${mon.mse_free}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>ModelOps</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Registry ID</div><div class="report-item-value">${modelOps.model_registry_id}</div></div>
        <div><div class="report-item-label">Training data version</div><div class="report-item-value">${modelOps.training_data_version}</div></div>
        <div><div class="report-item-label">Retraining trigger</div><div class="report-item-value">${modelOps.retraining_trigger}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Artifacts</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Model hash</div><div class="report-item-value">${artifacts.model_hash}</div></div>
        <div><div class="report-item-label">Concentration map hash</div><div class="report-item-value">${artifacts.concentration_map_hash}</div></div>
        <div><div class="report-item-label">Training data version</div><div class="report-item-value">${artifacts.training_data_version}</div></div>
      </div>
    </div>

    <div class="report-section">
      <h3>Security / Audit</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Bundle signature</div><div class="report-item-value">${bundle.signature}</div></div>
        <div><div class="report-item-label">Export signature</div><div class="report-item-value">${fexport.signature}</div></div>
        <div><div class="report-item-label">Compliance tags</div><div class="report-item-value">${(fexport.compliance_tags||[]).join(", ")}</div></div>
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

async function renderPdfMap() {
  return new Promise((resolve, reject) => {
    if (typeof leafletImage !== "function") {
      console.error("leafletImage non disponibile");
      return reject(new Error("leafletImage non disponibile"));
    }

    const pdfMap = L.map("pdf-map", {
      zoomControl: false,
      attributionControl: false,
      preferCanvas: true // <<< fondamentale
    });

    const canvasRenderer = L.canvas({ padding: 0.5 });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19
    }).addTo(pdfMap);

    // ---- SORGENTE (Circle su renderer Canvas) ----
    if (sourceCircle) {
      const sc = sourceCircle.getLatLng();

      L.circle(sc, {
        radius: 250,
        color: "#f97316",
        fillColor: "#fb923c",
        fillOpacity: 0.25,
        weight: 2,
        renderer: canvasRenderer
      }).addTo(pdfMap);
    }

    // ---- PERCORSO VAN (Polyline Canvas) ----
    if (pathLatLngs.length > 1) {
      L.polyline(pathLatLngs, {
        color: "#0ea5e9",
        weight: 3,
        renderer: canvasRenderer
      }).addTo(pdfMap);

      // marker finale del van
      L.marker(pathLatLngs[pathLatLngs.length - 1], {
        icon: vanIcon
      }).addTo(pdfMap);
    }

    // ---- FIT BOUNDS ----
    if (pathLatLngs.length > 1) {
      const bounds = L.latLngBounds(pathLatLngs);
      if (sourceCircle) bounds.extend(sourceCircle.getLatLng());
      pdfMap.fitBounds(bounds, { padding: [50, 50] });
    }

    // ---- ATTENDI IL RENDER COMPLETO ----
    setTimeout(() => {
      leafletImage(pdfMap, (err, canvas) => {
        pdfMap.remove();
        if (err) return reject(err);
        resolve(canvas.toDataURL("image/png"));
      });
    }, 800);
  });
}

async function exportPDF() {
  try {
    const JsPDF = getJsPDF();
    if (!JsPDF) {
      alert("Errore: libreria jsPDF non caricata.");
      console.error("window.jspdf o window.jspdf.jsPDF non disponibili");
      return;
    }

    const doc = new JsPDF({ unit: "pt", format: "a4" });

    const bundle = window.lastBundle;
    let monitoring = window.lastMonitoring;
    if (!bundle) {
      alert("Nessun forensic bundle disponibile. Esegui prima una simulazione con detection.");
      return;
    }

    const ev = bundle.event || {};
    const air = ev.SensorAir || {};
    const sub = ev.SensorSubstance || {};
    const gps = ev.SensorGPS || {};
    const piml = ev.PIML_Features || {};
    const inf = ev.Inference || {};
    const evMonitoring = ev.Monitoring || {};
    const modelOps = ev.ModelOps || {};
    const forensicExport = ev.ForensicExport || {};
    const artifacts = ev.artifacts || {};

    if (!monitoring) monitoring = evMonitoring || {};

    // ------ PAGE 1: MAP ------
    let mapData = null;
    try {
      mapData = await renderPdfMap();
    } catch (e) {
      console.error("Errore nel renderPdfMap:", e);
    }

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();

    doc.setFontSize(20);
    doc.text("Forensic Detection Report – Map Overview", pageWidth / 2, 40, { align: "center" });

    if (mapData) {
      const mapW = pageWidth - 80;
      const mapH = mapW * 0.66;
      doc.addImage(mapData, "PNG", 40, 60, mapW, mapH);
    } else {
      doc.setFontSize(12);
      doc.text("Map snapshot not available.", 40, 80);
    }

    // ------ PAGE 2+: REPORT ------
    doc.addPage();
    let y = 40;

    const checkPage = (extra = 40) => {
      if (y + extra > pageHeight - 40) {
        doc.addPage();
        y = 40;
      }
    };

    const section = (title) => {
      checkPage(60);
      doc.setFontSize(16);
      doc.text(title, 40, y);
      y += 10;
      doc.setLineWidth(0.5);
      doc.line(40, y, pageWidth - 40, y);
      y += 20;
    };

    const field = (name, val) => {
      checkPage(30);

      const value = (val !== undefined && val !== null) ? String(val) : "N/A";

      const labelX = 40;
      const valueX = 220;   // ← spostato più a destra per leggibilità
      const maxWidth = pageWidth - valueX - 40;

      doc.setFontSize(12);
      doc.text(name + ":", labelX, y);

      const wrapped = doc.splitTextToSize(value, maxWidth);
      doc.text(wrapped, valueX, y);

      y += wrapped.length * 14;
      y += 6;
    };


    // Event metadata
    section("Event metadata");
    field("Simulation ID", ev.simulation_id);
    field("Timestamp", ev.timestamp);
    field("Bundle name", bundle.bundle_name);
    field("SHA256", bundle.hash_sha256);

    // Environmental
    section("Environmental conditions");
    field("Temperature (°C)", air.temperature_C);
    field("Humidity (%)", air["humidity_%"]);
    field("Wind", `${air.wind_speed_mps} m/s @ ${air.wind_dir_deg}°`);
    field("Stability class", air.stability_class);

    // Substance
    section("Detected substance");
    field("Compound", sub.compound_name);
    field("Formula", sub.molecular_formula);
    field("Noise", sub.noise_level);
    field("EI Mass Spectrum (600 bins)", (sub.spectrum_ei_1_600 || []).join(", "));

    // PIML
    section("PIML features");
    field("Sigma_y", piml.sigma_y);
    field("Sigma_z", piml.sigma_z);
    field("Péclet number", piml.pe_number);
    field("Stability index", piml.stability_index);

    // Location
    section("Location");
    field("Latitude", gps.latitude);
    field("Longitude", gps.longitude);
    field("Altitude", gps.altitude_m);

    // Inference
    section("Inference");
    field("Predicted class", inf.predicted_class);
    field("Confidence", inf.confidence_score);
    field("Dispersion map ID", inf.dispersion_map_id);

    // Monitoring
    section("Monitoring");
    field("Model version", monitoring.model_version);
    field("Latency (ms)", monitoring.latency_ms);
    field("Drift score", monitoring.drift_score);
    field("MSE free", monitoring.mse_free);

    // ModelOps
    section("ModelOps");
    field("Registry ID", modelOps.model_registry_id);
    field("Training data version", modelOps.training_data_version);
    field("Retraining trigger", modelOps.retraining_trigger);

    // Artifacts
    section("Artifacts");
    field("Model hash", artifacts.model_hash);
    field("Concentration map hash", artifacts.concentration_map_hash);
    field("Training version", artifacts.training_data_version);

    // Audit
    section("Security & Audit");
    field("Bundle signature", bundle.signature);
    field("Export signature", forensicExport.signature);
    field("Compliance tags", (forensicExport.compliance_tags || []).join(", "));

    const simId = ev.simulation_id || "unknown";
    doc.save(`forensic_report_${simId}.pdf`);
  } catch (err) {
    console.error("Errore nella generazione del PDF:", err);
    alert("Errore durante la generazione del PDF. Controlla la console del browser per i dettagli.");
  }
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

      resetGraphics();
      window.lastBundle = null;
      window.lastMonitoring = null;

      simIdEl.textContent = msg.simulation_id || "–";
      setStatus("patrolling");

      btnStart.disabled = true;
      btnDebug.disabled = true;
      btnReset.disabled = false;

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

      vanMarker = L.marker([vLat, vLon], {
        icon: vanIcon
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
      btnStart.disabled = true;
      btnDebug.disabled = true;
      btnReset.disabled = false;

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
      btnDebug.disabled = false;   // ora puoi ricominciare
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
      btnDebug.disabled = false;
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
  
  // RESET COMPLETO PRIMA DI RIPARTIRE
  resetGraphics();
  hideSimCard();
  hideBundleCard();
  hideProcessing();
  window.lastBundle = null;
  window.lastMonitoring = null;

  btnStart.disabled = true;
  btnDebug.disabled = true;
  btnReset.disabled = true;

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
  window.lastBundle = null;
  window.lastMonitoring = null;

});

ensureMap();
hideLoading();
connectWebSocket();