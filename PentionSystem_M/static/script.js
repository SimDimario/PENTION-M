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
//const simDistEl = document.getElementById("sim-distance");
const modelVersionEl = document.getElementById("model-version");
const driftScoreEl = document.getElementById("drift-score");
const latencyEl = document.getElementById("latency-ms");
const bundleNameEl = document.getElementById("bundle-name");
const bundleLogEl = document.getElementById("bundle-log");
const monitoringLogEl = document.getElementById("monitoring-log");
const processingCard = document.getElementById("processing-card");
const simCard = document.getElementById("sim-card");
const bundleCard = document.getElementById("bundle-card");
const monitorCard = document.getElementById("monitor-card");

function showSimCard() { simCard.style.display = "block"; }
function hideSimCard() { simCard.style.display = "none"; }

function showBundleCard() { bundleCard.style.display = "block"; }
function hideBundleCard() { bundleCard.style.display = "none"; }

function showMonitorCard() { monitorCard.style.display = "block"; }
function hideMonitorCard() { monitorCard.style.display = "none"; }

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

      if (msg.simulation_id) {
        simIdEl.textContent = msg.simulation_id;
      }
      if (msg.monitoring) {
        monitoringLogEl.textContent = JSON.stringify(msg.monitoring, null, 2);
        showMonitorCard();

        if (msg.monitoring.model_version) {
          modelVersionEl.textContent = msg.monitoring.model_version;
        }
        if (msg.monitoring.latency_ms !== undefined) {
          latencyEl.textContent = msg.monitoring.latency_ms + " ms";
        }
        if (msg.monitoring.drift_score !== undefined) {
          driftScoreEl.textContent = msg.monitoring.drift_score;
        }
      }

      if (msg.registry && msg.registry.current_model_version) {
        modelVersionEl.textContent = msg.registry.current_model_version;
      }

      if (msg.forensic_bundle) {
        bundleNameEl.textContent = msg.forensic_bundle.bundle_name || "bundle";
        bundleLogEl.textContent = JSON.stringify(msg.forensic_bundle, null, 2);
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
  hideMonitorCard();
});

ensureMap();
connectWebSocket();
hideLoading();
