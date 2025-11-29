let map;
let vanMarker = null;
let sourceCircle = null;
let vanPath = null;
let pathLatLngs = [];
let ws = null;
let vanIcon = null;
let miniSpectrumChart = null;
let fullSpectrumChart = null;

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

  // === RESET MASS SPECTRUM CHARTS ===
  if (miniSpectrumChart) {
      miniSpectrumChart.destroy();
      miniSpectrumChart = null;
  }
  if (fullSpectrumChart) {
      fullSpectrumChart.destroy();
      fullSpectrumChart = null;
  }

  // Nasconde il box mini-spectra
  document.getElementById("mini-spectrum-card").style.display = "none";

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
  const sourceGPS = ev.SourceGPS || {};
  const inferenceLatLon = ev.inference_latlon || {};
  const pimlRuntime = ev.PIML_Runtime || {};
  const sensorTS = ev.SensorNetworkTimeSeries || [];

  const mon = monitoring || evMonitoring;

  const content = document.getElementById("popup-content");

  // helper per sicurezza
  const safe = v => (v === undefined || v === null ? "N/A" : v);

  content.innerHTML = `
    <!-- EVENT METADATA -->
    <div class="report-section">
      <h3>Event metadata</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Simulation ID</div><div class="report-item-value">${safe(ev.simulation_id)}</div></div>
        <div><div class="report-item-label">Timestamp</div><div class="report-item-value">${safe(ev.timestamp)}</div></div>
        <div><div class="report-item-label">Bundle name</div><div class="report-item-value">${safe(bundle.bundle_name)}</div></div>
        <div><div class="report-item-label">Hash SHA256</div><div class="report-item-value">${safe(bundle.hash_sha256)}</div></div>
      </div>
    </div>

    <!-- PUBLIC KEY -->
    <div class="report-section">
      <h3>Public Key</h3>
      <div class="report-item-value" style="word-break: break-all;">
        ${safe(bundle.public_key)
            .replace("-----BEGIN PUBLIC KEY-----","")
            .replace("-----END PUBLIC KEY-----","")
            .trim()}
      </div>
    </div>


    <!-- ENVIRONMENT -->
    <div class="report-section">
      <h3>Environmental conditions</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Temperature (°C)</div><div class="report-item-value">${safe(air.temperature_C)}</div></div>
        <div><div class="report-item-label">Humidity (%)</div><div class="report-item-value">${safe(air["humidity_%"])}</div></div>
        <div><div class="report-item-label">Wind</div><div class="report-item-value">${safe(air.wind_speed_mps)} m/s @ ${safe(air.wind_dir_deg)}°</div></div>
        <div><div class="report-item-label">Stability class</div><div class="report-item-value">${safe(air.stability_class)}</div></div>
      </div>
    </div>

    <!-- SUBSTANCE -->
    <div class="report-section">
      <h3>Detected substance</h3>

      <div class="report-grid">
        <div><div class="report-item-label">Compound</div><div class="report-item-value">${safe(sub.compound_name)}</div></div>
        <div><div class="report-item-label">Noise level</div><div class="report-item-value">${safe(sub.noise_level)}</div></div>
      </div>

      <h3 style="margin-top:20px; margin-bottom:10px;">EI Mass Spectrum (600 bins)</h3>
      <canvas id="full-spectrum-chart" height="260"></canvas>
    </div>


    <!-- PIML FEATURES -->
    <div class="report-section">
      <h3>PIML features</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Sigma_y</div><div class="report-item-value">${safe(piml.sigma_y)}</div></div>
        <div><div class="report-item-label">Sigma_z</div><div class="report-item-value">${safe(piml.sigma_z)}</div></div>
        <div><div class="report-item-label">Péclet number</div><div class="report-item-value">${safe(piml.pe_number)}</div></div>
        <div><div class="report-item-label">Stability index</div><div class="report-item-value">${safe(piml.stability_index)}</div></div>
      </div>
    </div>

    <!-- SENSOR GPS -->
    <div class="report-section">
      <h3>Van location (GPS)</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Latitude</div><div class="report-item-value">${safe(gps.latitude)}</div></div>
        <div><div class="report-item-label">Longitude</div><div class="report-item-value">${safe(gps.longitude)}</div></div>
        <div><div class="report-item-label">Altitude (m)</div><div class="report-item-value">${safe(gps.altitude_m)}</div></div>
      </div>
    </div>

    <!-- SOURCE GPS -->
    <div class="report-section">
      <h3>Simulated source location</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Latitude</div><div class="report-item-value">${safe(sourceGPS.latitude)}</div></div>
        <div><div class="report-item-label">Longitude</div><div class="report-item-value">${safe(sourceGPS.longitude)}</div></div>
      </div>
    </div>

    <!-- INFERENCE -->
    <div class="report-section">
      <h3>Inference</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Predicted class</div><div class="report-item-value">${safe(inf.predicted_class)}</div></div>
        <div><div class="report-item-label">Confidence</div><div class="report-item-value">${safe(inf.confidence_score)}</div></div>
        <div><div class="report-item-label">Dispersion map ID</div><div class="report-item-value">${safe(inf.dispersion_map_id)}</div></div>
        <div><div class="report-item-label">Predicted source (x,y)</div><div class="report-item-value">${safe(inf.predicted_source_location)}</div></div>
        <div><div class="report-item-label">Predicted location (lat,lon)</div><div class="report-item-value">${safe(inferenceLatLon.latitude)}, ${safe(inferenceLatLon.longitude)}</div></div>
      </div>
    </div>

    <!-- MONITORING -->
    <div class="report-section">
      <h3>Monitoring</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Model version</div><div class="report-item-value">${safe(mon.model_version)}</div></div>
        <div><div class="report-item-label">Latency (ms)</div><div class="report-item-value">${safe(mon.latency_ms)}</div></div>
        <div><div class="report-item-label">Drift score</div><div class="report-item-value">${safe(mon.drift_score)}</div></div>
        <div><div class="report-item-label">MSE free</div><div class="report-item-value">${safe(mon.mse_free ?? ev.Monitoring?.mse_free)}</div></div>
      </div>
    </div>

    <!-- PIML RUNTIME -->
    <div class="report-section">
      <h3>PIML Runtime</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Correction – Status</div><div class="report-item-value">${safe(pimlRuntime.correction_dispersion_piml?.status)}</div></div>
        <div><div class="report-item-label">Correction – Model</div><div class="report-item-value">${safe(pimlRuntime.correction_dispersion_piml?.model_version)}</div></div>
        <div><div class="report-item-label">Correction – Shape</div><div class="report-item-value">${safe(pimlRuntime.correction_dispersion_piml?.corrected_map_shape)}</div></div>
        <div><div class="report-item-label">Correction – Hash</div><div class="report-item-value">${safe(pimlRuntime.correction_dispersion_piml?.corrected_map_hash)}</div></div>

        <div><div class="report-item-label">SourceLoc – Status</div><div class="report-item-value">${safe(pimlRuntime.source_localization_piml?.status)}</div></div>
        <div><div class="report-item-label">SourceLoc – Model</div><div class="report-item-value">${safe(pimlRuntime.source_localization_piml?.model_version)}</div></div>
        <div><div class="report-item-label">SourceLoc – XY</div><div class="report-item-value">${safe(pimlRuntime.source_localization_piml?.predicted_source_xy)}</div></div>
        <div><div class="report-item-label">SourceLoc – Confidence</div><div class="report-item-value">${safe(pimlRuntime.source_localization_piml?.confidence)}</div></div>
      </div>
    </div>

    <!-- MODELOPS -->
    <div class="report-section">
      <h3>ModelOps</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Registry ID</div><div class="report-item-value">${safe(modelOps.model_registry_id)}</div></div>
        <div><div class="report-item-label">Training data version</div><div class="report-item-value">${safe(modelOps.training_data_version)}</div></div>
        <div><div class="report-item-label">Retraining trigger</div><div class="report-item-value">${safe(modelOps.retraining_trigger)}</div></div>
      </div>
    </div>

    <!-- ARTIFACTS -->
    <div class="report-section">
      <h3>Artifacts</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Model hash</div><div class="report-item-value">${safe(artifacts.model_hash)}</div></div>
        <div><div class="report-item-label">Concentration map hash</div><div class="report-item-value">${safe(artifacts.concentration_map_hash)}</div></div>
        <div><div class="report-item-label">Training data version</div><div class="report-item-value">${safe(artifacts.training_data_version)}</div></div>
      </div>
    </div>

    <!-- SECURITY -->
    <div class="report-section">
      <h3>Security / Audit</h3>
      <div class="report-grid">
        <div><div class="report-item-label">Bundle hash</div><div class="report-item-value">${safe(bundle.hash_sha256)}</div></div>
        <div><div class="report-item-label">Digital signature</div><div class="report-item-value">${safe(bundle.signature)}</div></div>
        <div><div class="report-item-label">Compliance tags</div><div class="report-item-value">${(fexport.compliance_tags || []).join(", ")}</div></div>
      </div>
    </div>
  `;

  document.getElementById("report-popup").style.display = "flex";

  // Render full spectrum in popup
  const spectrum = sub.spectrum_ei_1_600 || null;
  if (spectrum) {
    fullSpectrumChart = renderMassSpectrum(
      "full-spectrum-chart",
      spectrum,
      { clearExisting: true, instanceRef: { value: fullSpectrumChart } }
    );
  }

}


function closeReportPopup() {

  if (fullSpectrumChart) {
      fullSpectrumChart.destroy();
      fullSpectrumChart = null;
  }

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

function renderMassSpectrum(canvasId, spectrum, options = {}) {
  const ctx = document.getElementById(canvasId)?.getContext("2d");
  if (!ctx) return;

  if (options.clearExisting && options.instanceRef) {
    if (options.instanceRef.value) {
      options.instanceRef.value.destroy();
    }
  }

  const labels = Array.from({ length: spectrum.length }, (_, i) => i + 1);

  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Mass spectrum",
        data: spectrum,
        backgroundColor: "rgba(56,189,248,0.85)",
        borderWidth: 0,
        barPercentage: 1.0,
        categoryPercentage: 1.0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { display: false },
          grid: { display: false }
        },
        y: {
          ticks: { display: false },
          grid: { display: false }
        }
      }
    }
  });

  return chart;
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
      // forza Leaflet a ricalcolare le dimensioni del canvas
      pdfMap.invalidateSize();

      leafletImage(pdfMap, (err, canvas) => {
        pdfMap.remove();
        if (err) return reject(err);
        resolve(canvas.toDataURL("image/png"));
      });
    }, 1200); // timeout leggermente più alto per sicurezza
  });
}

async function exportPDF() {
  try {
    const JsPDF = getJsPDF();
    if (!JsPDF) {
      alert("Errore: libreria jsPDF non caricata.");
      return;
    }

    const doc = new JsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();

    // ============================
    // COVER PAGE (PATCH 1)
    // ============================
    try {
        const logoEl = document.querySelector(".report-logo");
        if (logoEl) {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.src = logoEl.src;

            await new Promise(res => img.onload = res);

            // ridimensioniamo a risoluzione PDF *2 (retina-like)
            const canvas = document.createElement("canvas");
            canvas.width = img.width * 2;
            canvas.height = img.height * 2;

            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            const dataURL = canvas.toDataURL("image/png");

            doc.addImage(
                dataURL,
                "PNG",
                pageWidth / 2 - 60,   // centrato
                40,                   // top
                120,                  // width ad alta risoluzione
                (img.height / img.width) * 120 // proporzioni corrette
            );
        }
    } catch (e) {
        console.error("Logo cover error", e);
    }


    doc.setFontSize(22);
    doc.text("Forensic Detection Report", pageWidth/2, 150, { align: "center" });

    doc.setFontSize(14);
    doc.text("Amsterdam Mobile Lab – PIML & MLOps Pipeline", pageWidth/2, 170, { align: "center" });

    let y = 220;

    // carica dati bundle
    const bundle = window.lastBundle;
    let monitoring = window.lastMonitoring;
    if (!bundle) {
      alert("Nessun forensic bundle disponibile.");
      return;
    }

    const ev = bundle.event || {};
    const safe = v => (v === undefined || v === null ? "N/A" : v);
    const air = ev.SensorAir || {};
    const sub = ev.SensorSubstance || {};
    const gps = ev.SensorGPS || {};
    const piml = ev.PIML_Features || {};
    const inf = ev.Inference || {};
    const evMonitoring = ev.Monitoring || {};
    const modelOps = ev.ModelOps || {};
    const forensicExport = ev.ForensicExport || {};
    const artifacts = ev.artifacts || {};
    const pimlRuntime = ev.PIML_Runtime || {};

    if (!monitoring) monitoring = evMonitoring || {};

    // ============================
    // PAGE 1 — MAP
    // ============================
    let mapData = null;
    try {
      mapData = await renderPdfMap();
    } catch (e) {
      console.error("Errore nel renderPdfMap:", e);
    }

    doc.setFontSize(20);
    doc.text("Map Overview", pageWidth/2, y, { align: "center" });
    y += 20;

    if (mapData) {
      const mapW = pageWidth - 80;
      const mapH = mapW * 0.66;
      doc.addImage(mapData, "PNG", 40, y, mapW, mapH);
      y += mapH + 40;
    } else {
      doc.setFontSize(12);
      doc.text("Map snapshot not available.", 40, y);
      y += 40;
    }

    // ============================
    // PAGE 2 — FULL REPORT
    // ============================
    doc.addPage();
    y = 40;

    const checkPage = (extra = 40) => {
      if (y + extra > pageHeight - 40) {
        doc.addPage();
        y = 40;
      }
    };

    // === PATCH 2: Section elegante
    const section = (title) => {
      y += 14;
      checkPage(80);

      doc.setFontSize(16);
      doc.setTextColor(30);
      doc.text(title, 40, y);

      y += 12;

      doc.setDrawColor(170);
      doc.setLineWidth(0.7);
      doc.line(40, y, pageWidth - 40, y);

      y += 18;
      doc.setTextColor(0);
    };

    const field = (label, value) => {
      checkPage(30);
      const v = (value === undefined || value === null ? "N/A" : String(value));

      const labelX = 40;
      const valueX = 220;
      const maxWidth = pageWidth - valueX - 40;

      doc.setFontSize(12);
      doc.text(label + ":", labelX, y);

      const wrapped = doc.splitTextToSize(v, maxWidth);
      doc.text(wrapped, valueX, y);

      y += wrapped.length * 14 + 6;
    };

    // ====== FULL REPORT SECTIONS ======

    section("Event metadata");
    field("Simulation ID", ev.simulation_id);
    field("Timestamp", ev.timestamp);
    field("Bundle name", bundle.bundle_name);
    field("Bundle hash", bundle.hash_sha256);

    section("Public Key");
    field(
      "PEM",
      safe(bundle.public_key)
        .replace("-----BEGIN PUBLIC KEY-----","")
        .replace("-----END PUBLIC KEY-----","")
        .trim()
    );

    section("Environmental conditions");
    field("Temperature (°C)", air.temperature_C);
    field("Humidity (%)", air["humidity_%"]);
    field("Wind", `${air.wind_speed_mps} m/s @ ${air.wind_dir_deg}°`);
    field("Stability class", air.stability_class);

    section("Detected substance");
    field("Compound", sub.compound_name);
    field("Noise level", sub.noise_level);

    checkPage(260);
    doc.setFontSize(14);
    doc.text("EI Mass Spectrum (600 bins)", 40, y);
    y += 10;

    // === HIGH-RES MASS SPECTRUM PATCH ===
    try {
        const canvas = document.getElementById("full-spectrum-chart");
        if (canvas) {
            // Cattura il canvas in alta risoluzione
            const exportCanvas = document.createElement("canvas");
            exportCanvas.width = canvas.width * 2;
            exportCanvas.height = canvas.height * 2;

            const ctx = exportCanvas.getContext("2d");
            ctx.scale(2, 2);
            ctx.drawImage(canvas, 0, 0);

            const imgData = exportCanvas.toDataURL("image/png");

            const aspect = exportCanvas.height / exportCanvas.width;
            const width = pageWidth - 80;
            const height = width * aspect;

            doc.addImage(imgData, "PNG", 40, y, width, height);

            y += height + 20;
        }
    } catch (e) {
        console.error("Spectrum PDF error:", e);
    }


    section("PIML Features");
    field("Sigma_y", piml.sigma_y);
    field("Sigma_z", piml.sigma_z);
    field("Péclet number", piml.pe_number);
    field("Stability index", piml.stability_index);

    section("Van location (GPS)");
    field("Latitude", gps.latitude);
    field("Longitude", gps.longitude);
    field("Altitude (m)", gps.altitude_m);

    section("Simulated source location");
    field("Latitude", safe(ev.SourceGPS?.latitude));
    field("Longitude", safe(ev.SourceGPS?.longitude));

    section("Inference");
    field("Predicted class", inf.predicted_class);
    field("Confidence", inf.confidence_score);
    field("Dispersion map ID", inf.dispersion_map_id);
    field("Predicted source (x,y)", inf.predicted_source_location);
    field("Lat/Lon", `${safe(ev.inference_latlon?.latitude)}, ${safe(ev.inference_latlon?.longitude)}`);

    section("Monitoring");
    field("Model version", monitoring.model_version);
    field("Latency (ms)", monitoring.latency_ms);
    field("Drift score", monitoring.drift_score);
    field("MSE free", monitoring.mse_free);

    section("PIML Runtime");
    field("Correction status", pimlRuntime.correction_dispersion_piml?.status);
    field("Correction model", pimlRuntime.correction_dispersion_piml?.model_version);
    field("Shape", pimlRuntime.correction_dispersion_piml?.corrected_map_shape);
    field("Hash", pimlRuntime.correction_dispersion_piml?.corrected_map_hash);

    field("SourceLoc status", pimlRuntime.source_localization_piml?.status);
    field("SourceLoc model", pimlRuntime.source_localization_piml?.model_version);
    field("SourceLoc XY", pimlRuntime.source_localization_piml?.predicted_source_xy);
    field("SourceLoc confidence", pimlRuntime.source_localization_piml?.confidence);

    section("ModelOps");
    field("Registry ID", modelOps.model_registry_id);
    field("Training data version", modelOps.training_data_version);
    field("Retraining trigger", modelOps.retraining_trigger);

    section("Artifacts");
    field("Model hash", artifacts.model_hash);
    field("Concentration map hash", artifacts.concentration_map_hash);
    field("Training data version", artifacts.training_data_version);

    section("Security / Audit");
    field("Signature", bundle.signature);
    field("Compliance tags", (forensicExport.compliance_tags || []).join(", "));

    // ============================
    // PATCH 3 — PAGE NUMBERS
    // ============================
    function addPageNumber() {
      const pageCount = doc.internal.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i);
        doc.setFontSize(10);
        doc.setTextColor(120);
        doc.text(`Page ${i} / ${pageCount}`, pageWidth - 60, pageHeight - 20);
      }
    }
    addPageNumber();

    doc.save(`forensic_report_${ev.simulation_id || "simulation"}.pdf`);

  } catch (err) {
    console.error("Errore nella generazione del PDF:", err);
    alert("Errore durante la generazione del PDF.");
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

          // opzionale: mostra in secondi invece che in ms
          const latSec = (latVal / 1000).toFixed(1);
          latencyEl.textContent = `${latSec} s`;

          // soglie nuove, tarate sulla tua pipeline reale
          colorizeMetric(latencyEl, latVal, {
            yellow: 6000,  // > 6 s → giallo
            red: 10000     // > 10 s → rosso
          });
        }

        if (msg.monitoring.stability_index !== undefined) {
          const stab = msg.monitoring.stability_index;
          const stabInt = Math.round(stab);
          stabilityEl.textContent = stabInt;

          stabilityEl.dataset.tooltip = `
            Stability Index = ${stabInt}
            ---
            1 = Very Unstable
            2 = Moderately Unstable
            3 = Slightly Unstable
            4 = Neutral
            5 = Moderately Stable
            6 = Very Stable
          `.trim();

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

        // Render mini mass spectrum
        const spec = msg.forensic_bundle?.event?.SensorSubstance?.spectrum_ei_1_600 || null;
        if (spec) {
          document.getElementById("mini-spectrum-card").style.display = "block";

          miniSpectrumChart = renderMassSpectrum(
            "mini-spectrum-chart",
            spec,
            { clearExisting: true, instanceRef: { value: miniSpectrumChart } }
          );
        }

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