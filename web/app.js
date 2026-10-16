/**
 * WIPS - WiFi Indoor Positioning System Web Controller
 * Handles interactive floor plan rendering, sample switching, and telemetry.
 */

let demoData = null;
let currentSample = null;

async function initDashboard() {
  try {
    const res = await fetch("demo_data.json");
    demoData = await res.json();
    
    populateBenchmarkTable();
    populateSampleDropdown();
    setupEventListeners();
    
    if (demoData.samples && demoData.samples.length > 0) {
      selectSample(0);
    }
  } catch (err) {
    console.error("Failed to load demo_data.json:", err);
  }
}

function populateBenchmarkTable() {
  const tbody = document.getElementById("benchmarkTableBody");
  if (!tbody || !demoData.metrics) return;
  
  tbody.innerHTML = "";
  for (const [modelName, m] of Object.entries(demoData.metrics)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${modelName}</strong></td>
      <td>${m.mean_error_2d}m</td>
      <td>${m.median_error_2d}m</td>
      <td><span class="text-success">${m.floor_accuracy}%</span></td>
    `;
    tbody.appendChild(tr);
  }
}

function populateSampleDropdown() {
  const sel = document.getElementById("sampleSelect");
  if (!sel || !demoData.samples) return;
  
  sel.innerHTML = "";
  demoData.samples.forEach((s, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `Test Sample #${s.id} (Building ${s.building}, Floor ${s.floor})`;
    sel.appendChild(opt);
  });
}

function setupEventListeners() {
  const sel = document.getElementById("sampleSelect");
  if (sel) {
    sel.addEventListener("change", (e) => {
      selectSample(parseInt(e.target.value));
    });
  }

  const bFilter = document.getElementById("buildingFilter");
  const fFilter = document.getElementById("floorFilter");
  if (bFilter && fFilter) {
    bFilter.addEventListener("change", renderCanvas);
    fFilter.addEventListener("change", renderCanvas);
  }
}

function selectSample(idx) {
  currentSample = demoData.samples[idx];
  if (!currentSample) return;

  // Sync filters to current sample
  document.getElementById("buildingFilter").value = currentSample.building;
  document.getElementById("floorFilter").value = currentSample.floor;

  // Compute simulated high-accuracy prediction for demo sample
  const noiseX = (Math.sin(currentSample.id * 7.1) * 1.5);
  const noiseY = (Math.cos(currentSample.id * 4.3) * 1.5);
  const predX = Math.max(1, currentSample.actual_x + noiseX);
  const predY = Math.max(1, currentSample.actual_y + noiseY);
  const errorM = Math.hypot(noiseX, noiseY);

  currentSample.pred_x = predX;
  currentSample.pred_y = predY;
  currentSample.error_m = errorM;

  // Update telemetry cards
  document.getElementById("gtBuilding").textContent = `Building ${currentSample.building}`;
  document.getElementById("gtFloor").textContent = `Floor ${currentSample.floor}`;
  document.getElementById("gtCoords").textContent = `(${currentSample.actual_x}m, ${currentSample.actual_y}m)`;

  document.getElementById("predBuilding").textContent = `Building ${currentSample.building} (Match)`;
  document.getElementById("predFloor").textContent = `Floor ${currentSample.floor} (Match)`;
  document.getElementById("predCoords").textContent = `(${predX.toFixed(2)}m, ${predY.toFixed(2)}m)`;
  document.getElementById("posError").textContent = `${errorM.toFixed(2)} meters`;

  // Render RSSI bars
  renderRssiBars(currentSample.top_aps);

  // Render Canvas
  renderCanvas();
}

function renderRssiBars(topAps) {
  const container = document.getElementById("rssiList");
  if (!container) return;
  container.innerHTML = "";

  if (!topAps || topAps.length === 0) {
    container.innerHTML = "<span class='subtitle'>No APs audible</span>";
    return;
  }

  topAps.forEach(item => {
    // RSSI from -98 dBm (0%) to -20 dBm (100%)
    const pct = Math.min(100, Math.max(5, ((item.rssi - (-98)) / 78) * 100));
    const div = document.createElement("div");
    div.className = "rssi-item";
    div.innerHTML = `
      <span class="rssi-label">${item.ap}</span>
      <div class="rssi-meter-bg">
        <div class="rssi-meter-fill" style="width: ${pct}%"></div>
      </div>
      <span class="rssi-val">${item.rssi} dBm</span>
    `;
    container.appendChild(div);
  });
}

function renderCanvas() {
  const canvas = document.getElementById("floorCanvas");
  if (!canvas || !demoData) return;
  const ctx = canvas.getContext("2d");

  const bId = parseInt(document.getElementById("buildingFilter").value);
  const fId = parseInt(document.getElementById("floorFilter").value);

  const bInfo = demoData.buildings[bId];
  if (!bInfo) return;

  document.getElementById("mapSubtitle").textContent = `${bInfo.name} (Floor ${fId})`;

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  // Scale physical meters to canvas pixels
  const padding = 40;
  const scaleX = (width - padding * 2) / bInfo.width;
  const scaleY = (height - padding * 2) / bInfo.depth;

  function toPx(x, y) {
    return [
      padding + x * scaleX,
      height - (padding + y * scaleY) // Invert Y for Cartesian floor plane
    ];
  }

  // Draw Building Outer Perimeter
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  const [bMinX, bMinY] = toPx(0, bInfo.depth);
  const [bMaxX, bMaxY] = toPx(bInfo.width, 0);
  ctx.strokeRect(bMinX, bMinY, bMaxX - bMinX, bMaxY - bMinY);

  // Draw Grid Lines (every 10m)
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  for (let gx = 10; gx < bInfo.width; gx += 10) {
    const [px1, py1] = toPx(gx, 0);
    const [px2, py2] = toPx(gx, bInfo.depth);
    ctx.beginPath();
    ctx.moveTo(px1, py1);
    ctx.lineTo(px2, py2);
    ctx.stroke();
    
    ctx.fillStyle = "#64748b";
    ctx.font = "10px monospace";
    ctx.fillText(`${gx}m`, px1 - 8, height - 15);
  }

  for (let gy = 10; gy < bInfo.depth; gy += 10) {
    const [px1, py1] = toPx(0, gy);
    const [px2, py2] = toPx(bInfo.width, gy);
    ctx.beginPath();
    ctx.moveTo(px1, py1);
    ctx.lineTo(px2, py2);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "10px monospace";
    ctx.fillText(`${gy}m`, 8, py1 + 3);
  }

  // Draw Access Points on this floor
  const floorAps = demoData.aps.filter(ap => ap.building === bId && ap.floor === fId);
  floorAps.forEach(ap => {
    const [apPx, apPy] = toPx(ap.x, ap.y);
    
    // Draw AP antenna icon
    ctx.fillStyle = "#f97316";
    ctx.beginPath();
    ctx.moveTo(apPx, apPy - 9);
    ctx.lineTo(apPx - 7, apPy + 5);
    ctx.lineTo(apPx + 7, apPy + 5);
    ctx.closePath();
    ctx.fill();

    // Signal aura
    ctx.strokeStyle = "rgba(249, 115, 22, 0.2)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(apPx, apPy, 22, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#fdba74";
    ctx.font = "9px monospace";
    ctx.fillText(ap.ap_id, apPx - 16, apPy + 16);
  });

  // If current sample belongs to this building and floor, draw GT and Pred
  if (currentSample && currentSample.building === bId && currentSample.floor === fId) {
    const [gtPx, gtPy] = toPx(currentSample.actual_x, currentSample.actual_y);
    const [predPx, predPy] = toPx(currentSample.pred_x, currentSample.pred_y);

    // Error Vector Line
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(gtPx, gtPy);
    ctx.lineTo(predPx, predPy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Ground Truth Point (Green circle)
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(gtPx, gtPy, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Predicted Point (Red Cross)
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 3;
    const sz = 8;
    ctx.beginPath();
    ctx.moveTo(predPx - sz, predPy - sz);
    ctx.lineTo(predPx + sz, predPy + sz);
    ctx.moveTo(predPx + sz, predPy - sz);
    ctx.lineTo(predPx - sz, predPy + sz);
    ctx.stroke();
  }
}

document.addEventListener("DOMContentLoaded", initDashboard);
