/**
 * AeroWing AI Pro - 3D Interactive Aerospace Telemetry & WebGL Engine
 */

let scene, camera, renderer, controls;
let wingMeshRight, wingMeshLeft, gridHelper;
let isSymmetric = true;
let isWireframe = false;
let currentTelemetry = null;
let debounceTimer = null;

// Initialize when DOM loads
document.addEventListener("DOMContentLoaded", () => {
  initThreeJS();
  initEventListeners();
  initTabs();
  // Initial evaluation
  triggerEvaluation();
});

function initThreeJS() {
  const container = document.getElementById("threejs-container");
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 500;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x060911);

  // Camera
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(30, 25, 45);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  container.appendChild(renderer.domElement);

  // OrbitControls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(10, 0, 0);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0x00f0ff, 0.8);
  dirLight1.position.set(20, 40, 20);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
  dirLight2.position.set(-20, -20, -20);
  scene.add(dirLight2);

  // Aerospace Coordinate Grid
  gridHelper = new THREE.GridHelper(80, 40, 0x00f0ff, 0x122240);
  gridHelper.position.y = -5;
  scene.add(gridHelper);

  // Animation Loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Resize Handler
  window.addEventListener("resize", () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

function getColormapColor(cp) {
  // Colormap: Cp from -1.8 (suction blue/magenta) to +0.8 (pressure red/orange)
  // Normalized t in [0, 1]
  const cpMin = -1.8;
  const cpMax = 0.8;
  const t = Math.max(0, Math.min(1, (cp - cpMin) / (cpMax - cpMin)));

  // Jet/Turbo gradient interpolation
  let r, g, b;
  if (t < 0.25) {
    r = 0;
    g = 4 * t;
    b = 1;
  } else if (t < 0.5) {
    r = 0;
    g = 1;
    b = 1 - 4 * (t - 0.25);
  } else if (t < 0.75) {
    r = 4 * (t - 0.5);
    g = 1;
    b = 0;
  } else {
    r = 1;
    g = 1 - 4 * (t - 0.75);
    b = 0;
  }
  return new THREE.Color(r, g, b);
}

function buildWingGeometry(meshData, isLeft = false) {
  const Xu = meshData.X_upper;
  const Yu = meshData.Y_upper;
  const Zu = meshData.Z_upper;
  const Xl = meshData.X_lower;
  const Yl = meshData.Y_lower;
  const Zl = meshData.Z_lower;
  const Cpu = meshData.Cp_upper;
  const Cpl = meshData.Cp_lower;

  const ny = Xu.length;
  const nx = Xu[0].length;

  const geometry = new THREE.BufferGeometry();
  const positions = [];
  const colors = [];
  const indices = [];

  const ySign = isLeft ? -1.0 : 1.0;

  // 1. Upper Surface Vertices
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      positions.push(Xu[j][i], Zu[j][i], Yu[j][i] * ySign);
      const col = getColormapColor(Cpu[j][i]);
      colors.push(col.r, col.g, col.b);
    }
  }

  // 2. Lower Surface Vertices
  const offsetLower = positions.length / 3;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      positions.push(Xl[j][i], Zl[j][i], Yl[j][i] * ySign);
      const col = getColormapColor(Cpl[j][i]);
      colors.push(col.r, col.g, col.b);
    }
  }

  // 3. Construct Triangles
  // Upper
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const p00 = j * nx + i;
      const p01 = j * nx + (i + 1);
      const p10 = (j + 1) * nx + i;
      const p11 = (j + 1) * nx + (i + 1);

      if (!isLeft) {
        indices.push(p00, p01, p11);
        indices.push(p00, p11, p10);
      } else {
        indices.push(p00, p11, p01);
        indices.push(p00, p10, p11);
      }
    }
  }

  // Lower
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const p00 = offsetLower + j * nx + i;
      const p01 = offsetLower + j * nx + (i + 1);
      const p10 = offsetLower + (j + 1) * nx + i;
      const p11 = offsetLower + (j + 1) * nx + (i + 1);

      if (!isLeft) {
        indices.push(p00, p11, p01);
        indices.push(p00, p10, p11);
      } else {
        indices.push(p00, p01, p11);
        indices.push(p00, p11, p10);
      }
    }
  }

  // Tip Cap (closing the wing tip at j = ny - 1)
  const jTip = ny - 1;
  for (let i = 0; i < nx - 1; i++) {
    const pu1 = jTip * nx + i;
    const pu2 = jTip * nx + (i + 1);
    const pl1 = offsetLower + jTip * nx + i;
    const pl2 = offsetLower + jTip * nx + (i + 1);

    if (!isLeft) {
      indices.push(pu1, pu2, pl2);
      indices.push(pu1, pl2, pl1);
    } else {
      indices.push(pu1, pl2, pu2);
      indices.push(pu1, pl1, pl2);
    }
  }

  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return geometry;
}

function update3DScene(meshData) {
  // Remove existing meshes
  if (wingMeshRight) scene.remove(wingMeshRight);
  if (wingMeshLeft) scene.remove(wingMeshLeft);

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.35,
    metalness: 0.25,
    wireframe: isWireframe,
    side: THREE.DoubleSide,
  });

  // Right Wing
  const geomRight = buildWingGeometry(meshData, false);
  wingMeshRight = new THREE.Mesh(geomRight, material);
  scene.add(wingMeshRight);

  // Left Wing (Symmetry)
  if (isSymmetric) {
    const geomLeft = buildWingGeometry(meshData, true);
    wingMeshLeft = new THREE.Mesh(geomLeft, material);
    scene.add(wingMeshLeft);
  }
}

async function triggerEvaluation() {
  const payload = {
    span: parseFloat(document.getElementById("sliderSpan").value),
    aspect_ratio: parseFloat(document.getElementById("sliderAR").value),
    taper_ratio: parseFloat(document.getElementById("sliderTaper").value),
    sweep_le_deg: parseFloat(document.getElementById("sliderSweep").value),
    dihedral_deg: parseFloat(document.getElementById("sliderDihedral").value),
    twist_root_deg: 2.0,
    twist_tip_deg: parseFloat(document.getElementById("sliderTwist").value),
    root_tc: parseFloat(document.getElementById("sliderRootTc").value),
    tip_tc: 0.10,
    alpha_deg: parseFloat(document.getElementById("sliderAlpha").value),
    mach: parseFloat(document.getElementById("sliderMach").value),
    reynolds: 2.5e7,
  };

  try {
    const response = await fetch("/api/wing/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    currentTelemetry = data;

    updateHUD(data.telemetry, data.geometry);
    update3DScene(data.mesh);
    drawLiftDistribution(data.telemetry.etas, data.telemetry.cl_spanwise, data.telemetry.cl);
  } catch (err) {
    console.error("Evaluation error:", err);
  }
}

function updateHUD(tele, geom) {
  // Telemetry Cards
  document.getElementById("teleCL").innerText = tele.cl.toFixed(3);
  document.getElementById("teleCD").innerText = tele.cd.toFixed(4);
  document.getElementById("teleCDCounts").innerText = (tele.cd * 10000).toFixed(1) + " drag counts";
  document.getElementById("teleLD").innerText = tele.l_over_d.toFixed(2);
  document.getElementById("barCL").style.width = Math.min(100, Math.max(0, tele.cl * 100)) + "%";

  document.getElementById("teleCDi").innerText = tele.cd_induced.toFixed(4);
  document.getElementById("teleCDp").innerText = tele.cd_profile.toFixed(4);
  document.getElementById("teleCDw").innerText = tele.cd_wave.toFixed(4);
  document.getElementById("teleSpanE").innerText = tele.span_efficiency.toFixed(3);
  document.getElementById("teleCM").innerText = tele.cm.toFixed(3);

  // Overlay HUD
  document.getElementById("hudSref").innerText = geom.s_ref.toFixed(2) + " m²";
  document.getElementById("hudMAC").innerText = geom.mac.toFixed(2) + " m";
  document.getElementById("hudRootC").innerText = geom.root_chord.toFixed(2) + " m";
  document.getElementById("hudFuel").innerText = geom.fuel_volume_m3.toFixed(2) + " m³";
}

function drawLiftDistribution(etas, clValues, totalCl) {
  const canvas = document.getElementById("liftCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Axes and Background Grid
  ctx.strokeStyle = "rgba(0, 240, 255, 0.1)";
  ctx.lineWidth = 1;
  for (let x = 50; x < w - 20; x += 80) {
    ctx.beginPath();
    ctx.moveTo(x, 20);
    ctx.lineTo(x, h - 30);
    ctx.stroke();
  }

  // Draw Ideal Elliptic Lift Distribution: cl_ideal(eta) = (4/pi) * CL * sqrt(1 - eta^2)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.setLineDash([5, 5]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 50; i++) {
    const eta = i / 50;
    const clIdeal = (4 / Math.PI) * totalCl * Math.sqrt(Math.max(0, 1 - eta * eta));
    const px = 50 + eta * (w - 80);
    const py = (h - 30) - (clIdeal / 1.2) * (h - 60);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw Actual Sectional Lift Distribution C_L(y)
  ctx.strokeStyle = "#00f0ff";
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let i = 0; i < etas.length; i++) {
    const eta = etas[i];
    const cl = clValues[i];
    const px = 50 + eta * (w - 80);
    const py = (h - 30) - (cl / 1.2) * (h - 60);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#8b9bb4";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.fillText("Root (η=0)", 50, h - 10);
  ctx.fillText("Tip (η=1.0)", w - 80, h - 10);
  ctx.fillText("— Actual 3D Lift C_L(y)", w - 240, 25);
  ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
  ctx.fillText("-- Ideal Elliptic", w - 120, 25);
}

function initEventListeners() {
  const sliders = [
    { id: "sliderSpan", label: "valSpan", unit: " m", fmt: (v) => parseFloat(v).toFixed(2) },
    { id: "sliderAR", label: "valAR", unit: "", fmt: (v) => parseFloat(v).toFixed(2) },
    { id: "sliderTaper", label: "valTaper", unit: "", fmt: (v) => parseFloat(v).toFixed(2) },
    { id: "sliderSweep", label: "valSweep", unit: "°", fmt: (v) => parseFloat(v).toFixed(1) },
    { id: "sliderDihedral", label: "valDihedral", unit: "°", fmt: (v) => parseFloat(v).toFixed(1) },
    { id: "sliderTwist", label: "valTwist", unit: "°", fmt: (v) => parseFloat(v).toFixed(1) },
    { id: "sliderRootTc", label: "valRootTc", unit: "%", fmt: (v) => (parseFloat(v) * 100).toFixed(1) },
    { id: "sliderAlpha", label: "valAlpha", unit: "°", fmt: (v) => parseFloat(v).toFixed(2) },
    { id: "sliderMach", label: "valMach", unit: "", fmt: (v) => parseFloat(v).toFixed(3) },
  ];

  sliders.forEach((s) => {
    const el = document.getElementById(s.id);
    const lbl = document.getElementById(s.label);
    if (!el) return;

    el.addEventListener("input", () => {
      lbl.innerText = s.fmt(el.value) + s.unit;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(triggerEvaluation, 35); // 35ms debouncing for instant response
    });
  });

  // Benchmark Selector
  document.getElementById("benchmarkSelect").addEventListener("change", async (e) => {
    const val = e.target.value;
    if (val === "custom") return;

    const res = await fetch(`/api/benchmark/${val}`);
    const bData = await res.json();

    document.getElementById("sliderSpan").value = bData.span;
    document.getElementById("valSpan").innerText = bData.span.toFixed(2) + " m";

    document.getElementById("sliderAR").value = bData.aspect_ratio;
    document.getElementById("valAR").innerText = bData.aspect_ratio.toFixed(2);

    document.getElementById("sliderTaper").value = bData.taper_ratio;
    document.getElementById("valTaper").innerText = bData.taper_ratio.toFixed(2);

    document.getElementById("sliderSweep").value = bData.sweep_le_deg;
    document.getElementById("valSweep").innerText = bData.sweep_le_deg.toFixed(1) + "°";

    document.getElementById("sliderDihedral").value = bData.dihedral_deg;
    document.getElementById("valDihedral").innerText = bData.dihedral_deg.toFixed(1) + "°";

    document.getElementById("sliderTwist").value = bData.twist_tip_deg;
    document.getElementById("valTwist").innerText = bData.twist_tip_deg.toFixed(1) + "°";

    document.getElementById("sliderMach").value = bData.recommended_flight.mach;
    document.getElementById("valMach").innerText = bData.recommended_flight.mach.toFixed(3);

    document.getElementById("sliderAlpha").value = bData.recommended_flight.alpha_deg;
    document.getElementById("valAlpha").innerText = bData.recommended_flight.alpha_deg.toFixed(2) + "°";

    triggerEvaluation();
  });

  // Symmetry & Wireframe buttons
  document.getElementById("btnToggleSymmetry").addEventListener("click", (e) => {
    isSymmetric = !isSymmetric;
    e.currentTarget.classList.toggle("active", isSymmetric);
    if (currentTelemetry) update3DScene(currentTelemetry.mesh);
  });

  document.getElementById("btnToggleWireframe").addEventListener("click", (e) => {
    isWireframe = !isWireframe;
    e.currentTarget.classList.toggle("active", isWireframe);
    if (currentTelemetry) update3DScene(currentTelemetry.mesh);
  });

  document.getElementById("btnResetCamera").addEventListener("click", () => {
    camera.position.set(30, 25, 45);
    controls.target.set(10, 0, 0);
  });

  // Inverse Design Execution
  document.getElementById("btnExecuteInverse").addEventListener("click", async () => {
    const cl = parseFloat(document.getElementById("targetClInput").value);
    const mach = parseFloat(document.getElementById("targetMachInput").value);
    const ar = parseFloat(document.getElementById("targetARInput").value);
    const ld = parseFloat(document.getElementById("targetLDInput").value);

    const res = await fetch("/api/wing/inverse-design", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_cl: cl, target_mach: mach, target_ar: ar, target_l_over_d: ld }),
    });
    const data = await res.json();
    const p = data.synthesized_parameters;

    const resBox = document.getElementById("inverseResultsBox");
    resBox.innerHTML = `
      <h3>SYNTHESIZED 3D WING SPECIFICATION</h3>
      <div style="font-family: ui-monospace, 'Cascadia Code', Consolas, monospace; font-size: 0.75rem; line-height: 1.6;">
        <div>Wingspan: <strong>${p.span} m</strong> | Aspect Ratio: <strong>${p.aspect_ratio}</strong></div>
        <div>Taper: <strong>${p.taper_ratio}</strong> | Sweep: <strong>${p.sweep_le_deg}°</strong></div>
        <div>Tip Washout: <strong>${p.twist_tip_deg}°</strong> | Dihedral: <strong>${p.dihedral_deg}°</strong></div>
      </div>
      <button id="btnApplyInverse" class="primary-btn" style="margin-top: 8px;">APPLY TO 3D VIEWPORT</button>
    `;

    document.getElementById("btnApplyInverse").addEventListener("click", () => {
      document.getElementById("sliderSpan").value = p.span;
      document.getElementById("valSpan").innerText = p.span + " m";
      document.getElementById("sliderAR").value = p.aspect_ratio;
      document.getElementById("valAR").innerText = p.aspect_ratio;
      document.getElementById("sliderTaper").value = p.taper_ratio;
      document.getElementById("valTaper").innerText = p.taper_ratio;
      document.getElementById("sliderSweep").value = p.sweep_le_deg;
      document.getElementById("valSweep").innerText = p.sweep_le_deg + "°";
      document.getElementById("sliderTwist").value = p.twist_tip_deg;
      document.getElementById("valTwist").innerText = p.twist_tip_deg + "°";
      triggerEvaluation();
    });
  });

  // Export buttons
  document.querySelectorAll(".export-dl-btn, #btnExportSTL").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const fmt = e.currentTarget.dataset.format || "stl";
      const payload = {
        span: parseFloat(document.getElementById("sliderSpan").value),
        aspect_ratio: parseFloat(document.getElementById("sliderAR").value),
        taper_ratio: parseFloat(document.getElementById("sliderTaper").value),
        sweep_le_deg: parseFloat(document.getElementById("sliderSweep").value),
        dihedral_deg: parseFloat(document.getElementById("sliderDihedral").value),
        twist_root_deg: 2.0,
        twist_tip_deg: parseFloat(document.getElementById("sliderTwist").value),
        root_tc: parseFloat(document.getElementById("sliderRootTc").value),
        tip_tc: 0.10,
        alpha_deg: parseFloat(document.getElementById("sliderAlpha").value),
        mach: parseFloat(document.getElementById("sliderMach").value),
        reynolds: 2.5e7,
      };

      const response = await fetch(`/api/export/file?export_format=${fmt}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `AeroWing_${fmt.toUpperCase()}.${fmt}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  });
}

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.dataset.tab;
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add("active");
    });
  });
}
