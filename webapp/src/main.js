/**
 * JemmaBrain main.js — Three.js cortical BOLD visualizer
 *
 * Data flow:
 *   /mesh/brain.glb    → GLTFLoader → BufferGeometry (position + normal)
 *   /api/networks      → Int16Array → network BufferAttribute (Yeo-7 index)
 *   /ws/bold           → binary frames + JSON messages → BOLD animation
 *   /api/result/latest → pre-loaded result on page refresh
 *
 * Visualization layers:
 *   1. BOLD heatmap shader  (inferno / plasma / turbo / coolwarm / viridis)
 *   2. Animated vertex pulse bloom at high-activation vertices
 *   3. Rim glow at silhouette edge
 *   4. Yeo-7 network overlay mode
 *   5. Hemisphere tint mode
 *   6. 3D CSS region labels
 *   7. Network activity sparkline chart
 *   8. Haemodynamic response curve
 *   9. 5-second segment breakdown with dominant network
 */

import * as THREE           from 'three'
import { OrbitControls }    from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader }       from 'three/addons/loaders/GLTFLoader.js'
import Stats                from 'stats.js'
import { gsap }             from 'gsap'

import {
  createBOLDMaterial,
  uploadBOLDFrame,
  YEO7_COLOURS,
  COLOUR_MAPS,
  COLOURMAP_NAMES,
  buildLegendGradient,
} from './BOLDShader.js'

import { showRecipeIntro } from './intro.js'

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const TR_SECONDS   = 0.5          // TRIBE v2 samples at 2 Hz
const TRS_PER_SEG  = 10           // 10 TRs = 5 seconds per segment
const MESH_URL     = '/mesh/brain.glb'
const NETWORK_URL  = '/api/networks'
const WS_URL       = `ws://${location.host}/ws/bold`

const YEO7_NAMES = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
const YEO7_HEX   = ['#781286', '#4682B4', '#00760E', '#C43AFA', '#DCF8A4', '#E69422', '#DC143C']

// Approximate MNI positions of major sulci/gyri for 3D labels (in mesh coords)
// These are set after mesh centring and will be overridden by server if available
const REGION_LABELS_DEFAULT = [
  { name: 'V1',    pos: new THREE.Vector3(-40,  -80, 10) },
  { name: 'M1',    pos: new THREE.Vector3(-35,  -20, 55) },
  { name: 'PFC',   pos: new THREE.Vector3(-25,   55, 20) },
  { name: 'IPL',   pos: new THREE.Vector3(-50,  -50, 40) },
  { name: 'insula',pos: new THREE.Vector3(-40,    5,  5) },
  { name: 'SMA',   pos: new THREE.Vector3(  0,  -10, 60) },
  { name: 'ACC',   pos: new THREE.Vector3(  0,   25, 30) },
  { name: 'PCC',   pos: new THREE.Vector3(  0,  -55, 28) },
]

// ─────────────────────────────────────────────────────────────────────────────
// DOM refs
// ─────────────────────────────────────────────────────────────────────────────
const $container    = document.getElementById('canvas-container')
const $loading      = document.getElementById('loading')
const $loadText     = document.getElementById('loading-text')
const $loadStage    = document.getElementById('loading-stage')
const $loadBar      = document.getElementById('loading-bar')
const $play         = document.getElementById('tl-play')
const $slider       = document.getElementById('tl-slider')
const $tlTime       = document.getElementById('tl-time')
const $tlTicks      = document.getElementById('tl-ticks')
const $speedBtns    = document.querySelectorAll('.tl-speed')
const $tierBtns     = document.querySelectorAll('.tier-btn')
const $narration    = document.getElementById('narration-text')
const $stimTitle    = document.getElementById('stimulus-title')
const $stimMeta     = document.getElementById('stimulus-meta')
const $peakTime     = document.getElementById('peak-time')
const $peakRegion   = document.getElementById('peak-region')
const $domNet       = document.getElementById('dominant-net')
const $domSub       = document.getElementById('dominant-sub')
const $cortexPct    = document.getElementById('cortex-pct')
const $currentTR    = document.getElementById('current-tr')
const $netList      = document.getElementById('network-list')
const $segList      = document.getElementById('seg-list')
const $roiList      = document.getElementById('roi-list')
const $tooltip      = document.getElementById('roi-tooltip')
const $labelCont    = document.getElementById('label-container')
const $legendBar    = document.getElementById('legend-bar')
const $legendMapName= document.getElementById('legend-map-name')
const $legendLo     = document.getElementById('legend-lo')
const $legendHi     = document.getElementById('legend-hi')
const $actCanvas    = document.getElementById('activity-canvas')
const $hrfCanvas    = document.getElementById('hrf-canvas')

// Mode buttons
const $btnBold    = document.getElementById('btn-mode-bold')
const $btnNet     = document.getElementById('btn-mode-net')
const $btnHemi    = document.getElementById('btn-mode-hemi')
const $btnLabels  = document.getElementById('btn-labels')
const $btnSpin    = document.getElementById('btn-autorotate')

// ─────────────────────────────────────────────────────────────────────────────
// Three.js scene
// ─────────────────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({
  antialias:        true,
  alpha:            false,
  powerPreference:  'high-performance',  // forces discrete GPU (RTX 5090)
  precision:        'highp',             // full float32 precision in shaders
  logarithmicDepthBuffer: false,         // not needed for brain; saves fill rate
})
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.outputColorSpace      = THREE.SRGBColorSpace
renderer.toneMapping           = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure   = 1.20   // slightly brighter to compensate PBR energy conservation
renderer.shadowMap.enabled     = false  // no shadow maps (brain is self-lit)
$container.appendChild(renderer.domElement)

// ── WebGPU advisory ───────────────────────────────────────────────────────────
;(async () => {
  let gpuBadge = ''
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      if (adapter) {
        const info = adapter.info ?? {}
        const vendor = info.vendor ?? 'GPU'
        gpuBadge = `⚡ WebGPU: ${vendor}`
      }
    } catch { /* ignore */ }
  }
  if (!gpuBadge) {
    gpuBadge = '🖥 WebGL2 (WebGPU not enabled — enable in chrome://flags for best perf)'
  }
  const badge = document.createElement('div')
  badge.textContent = gpuBadge
  badge.style.cssText = [
    'position:fixed', 'top:4px', 'right:8px', 'z-index:500',
    'font-size:10px', 'color:rgba(255,255,255,0.45)', 'pointer-events:none',
  ].join(';')
  document.body.appendChild(badge)
})()

// ── PBR quality controls (keyboard) ──────────────────────────────────────────
// Q/W cycle roughness; E/R cycle SSS strength
let _pbrRoughness  = 0.45
let _pbrSSS        = 0.30
function _updatePBR() {
  if (!S.boldMat) return
  S.boldMat.uniforms.u_roughness.value    = _pbrRoughness
  S.boldMat.uniforms.u_sss_strength.value = _pbrSSS
}

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x070b12)
scene.fog = new THREE.FogExp2(0x070b12, 0.0012)

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.5, 1200)
camera.position.set(0, 20, 230)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping    = true
controls.dampingFactor    = 0.055
controls.rotateSpeed      = 0.55
controls.zoomSpeed        = 0.85
controls.minDistance      = 70
controls.maxDistance      = 500
controls.autoRotate       = true
controls.autoRotateSpeed  = 0.35
controls.target.set(0, 0, 0)

// Lights (scene-level, brain shader does its own diffuse internally)
const ambient = new THREE.AmbientLight(0x1a2040, 1.0)
scene.add(ambient)
const sun = new THREE.DirectionalLight(0xffffff, 0.5)
sun.position.set(80, 140, 100)
scene.add(sun)
const fill = new THREE.DirectionalLight(0x3040a0, 0.2)
fill.position.set(-80, -40, -60)
scene.add(fill)

// Stats
const stats = new Stats()
stats.showPanel(0)
Object.assign(stats.dom.style, { position:'fixed', top:'56px', left:'8px', zIndex:'200', opacity:'0.4' })
document.body.appendChild(stats.dom)

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
const S = {
  brainMesh:   null,     // THREE.Mesh
  boldMat:     null,     // ShaderMaterial
  glowMesh:    null,     // subtle additive shell
  nVertices:   0,

  // BOLD timeline
  boldFrames:  [],       // Float32Array[], indexed by TR
  nTRs:        0,
  currentTR:   0,
  prevTR:      0,
  playing:     false,
  playSpeed:   1,
  _lastTick:   0,
  _accumMs:    0,
  _dragging:   false,
  _startTime:  0,        // performance.now() when playback started

  // Colourmap
  colourMapIdx: 0,

  // Display mode: 0=BOLD, 1=Networks, 2=Hemisphere
  mode: 0,

  // Labels visible?
  labelsVisible: true,
  labelEls: [],          // DOM elements

  // Narrations by tier
  narrations: {},
  activeTier: 2,

  // Analysis data
  analysis: null,

  // Per-segment data: array of { start_s, end_s, peak_z, dominant_net, net_idx }
  segments: [],

  // Activity history: { netName → Float32Array of mean z per TR }
  activityHistory: {},

  // WS
  ws: null,
}

// ─────────────────────────────────────────────────────────────────────────────
// Raycaster
// ─────────────────────────────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster()
raycaster.params.Points = { threshold: 2 }
const mouse = new THREE.Vector2(-2, -2)
let hoveredVert = -1

renderer.domElement.addEventListener('mousemove', ev => {
  const r = renderer.domElement.getBoundingClientRect()
  mouse.x =  (ev.clientX - r.left) / r.width  * 2 - 1
  mouse.y = -((ev.clientY - r.top)  / r.height) * 2 + 1
  $tooltip.style.left = (ev.clientX + 16) + 'px'
  $tooltip.style.top  = (ev.clientY - 8)  + 'px'
})

renderer.domElement.addEventListener('click', () => {
  if (hoveredVert < 0) return
  const netIdx  = getNetworkAtVert(hoveredVert)
  const boldVal = getBoldAtVert(hoveredVert)
  const netName = netIdx >= 0 ? YEO7_NAMES[netIdx] : 'Unknown'
  flashNarration(`
    <b>Vertex ${hoveredVert}</b> — ${netName} network<br>
    BOLD z-score: <span style="color:var(--accent)">${boldVal !== null ? boldVal.toFixed(3) + 'σ' : '—'}</span><br>
    <span style="color:var(--text-dim);font-size:10px;">Click the network name above to highlight that network across the brain.</span>
  `)
})

function doRaycast() {
  if (!S.brainMesh) return
  raycaster.setFromCamera(mouse, camera)
  const hits = raycaster.intersectObject(S.brainMesh)
  if (hits.length) {
    const vi    = hits[0].face.a
    const netIdx = getNetworkAtVert(vi)
    const bold   = getBoldAtVert(vi)
    hoveredVert  = vi
    const netCol = netIdx >= 0 ? YEO7_HEX[netIdx] : '#888'
    const netName= netIdx >= 0 ? YEO7_NAMES[netIdx] : 'Unknown'
    $tooltip.innerHTML = `
      <span style="color:var(--text-dim)">vert ${vi}</span><br>
      <span style="color:${netCol}">⬤</span> ${netName}<br>
      BOLD: <span class="tt-value">${bold !== null ? bold.toFixed(2) + 'σ' : '—'}</span>
    `
    $tooltip.classList.add('visible')
  } else {
    hoveredVert = -1
    $tooltip.classList.remove('visible')
  }
}

function getNetworkAtVert(vi) {
  if (!S.brainMesh) return -1
  const a = S.brainMesh.geometry.getAttribute('network')
  return a ? Math.round(a.getX(vi)) : -1
}

function getBoldAtVert(vi) {
  if (!S.brainMesh) return null
  const a = S.brainMesh.geometry.getAttribute('bold')
  return a ? a.getX(vi) : null
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh loading
// ─────────────────────────────────────────────────────────────────────────────
async function loadBrainMesh() {
  setStage('Fetching brain mesh…', 15)
  const gltf = await new Promise((res, rej) =>
    new GLTFLoader().load(MESH_URL, res, prog => {
      if (prog.total) setProgress(15 + (prog.loaded / prog.total) * 35)
    }, rej)
  )

  let mesh = null
  gltf.scene.traverse(o => { if (o.isMesh && !mesh) mesh = o })
  if (!mesh) throw new Error('brain.glb contains no mesh')

  const geo = mesh.geometry
  if (!geo.getAttribute('normal')) geo.computeVertexNormals()
  S.nVertices = geo.getAttribute('position').count
  setStage(`Mesh loaded — ${S.nVertices.toLocaleString()} vertices`, 50)

  // Network labels
  setStage('Fetching Yeo-7 network labels…', 55)
  const netAttr = await fetchNetworkLabels(S.nVertices)
  geo.setAttribute('network', netAttr)

  // BOLD attributes (zeros)
  const zero = new Float32Array(S.nVertices)
  geo.setAttribute('bold',      new THREE.BufferAttribute(zero.slice(), 1))
  geo.setAttribute('bold_prev', new THREE.BufferAttribute(zero.slice(), 1))

  // BOLD shader — FRONT faces only (no DoubleSide → eliminates z-fighting clipping)
  S.boldMat = createBOLDMaterial()
  S.boldMat.side        = THREE.FrontSide
  S.boldMat.depthWrite  = true
  S.boldMat.renderOrder = 2     // renders last → on top of interior shell
  mesh.material = S.boldMat
  mesh.renderOrder = 2
  setStage('Centring mesh…', 70)

  // Centre + normalise scale
  geo.computeBoundingBox()
  const box = geo.boundingBox
  const ctr = new THREE.Vector3()
  box.getCenter(ctr)
  geo.translate(-ctr.x, -ctr.y, -ctr.z)
  const diag  = box.max.distanceTo(box.min)
  mesh.scale.setScalar(160 / diag)

  // ── Interior shell (back-face pass) — shows as dark glass inside ─────────
  // Rendered FIRST so front faces correctly occlude it. Prevents "holes"
  // showing inner geometry when sub-threshold verts are transparent.
  const interiorMat = new THREE.MeshStandardMaterial({
    color:       0x0a0e1c,
    roughness:   0.8,
    metalness:   0.0,
    transparent: true,
    opacity:     0.72,
    side:        THREE.BackSide,
    depthWrite:  false,
  })
  const interiorMesh = new THREE.Mesh(geo, interiorMat)
  interiorMesh.renderOrder = 0    // renders first
  S.brainMesh = mesh
  scene.add(interiorMesh)
  scene.add(mesh)
  setStage('Done', 100)

  // Fade in
  mesh.material.uniforms.u_alpha_global.value = 0
  interiorMat.opacity = 0
  gsap.to(mesh.material.uniforms.u_alpha_global, { value: 0.93, duration: 1.4, ease: 'power2.out' })
  gsap.to(interiorMat, { opacity: 0.72, duration: 1.4, ease: 'power2.out' })

  addGlowShell(mesh)
  addStarField()
}

async function fetchNetworkLabels(nVerts) {
  try {
    const r  = await fetch(NETWORK_URL)
    if (!r.ok) throw 0
    const buf = await r.arrayBuffer()
    const i16 = new Int16Array(buf)
    const f32 = new Float32Array(i16.length)
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i]
    return new THREE.BufferAttribute(f32, 1)
  } catch {
    console.warn('[JemmaBrain] /api/networks unavailable — all Unknown')
    return new THREE.BufferAttribute(new Float32Array(nVerts).fill(-1), 1)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Visual extras
// ─────────────────────────────────────────────────────────────────────────────
function addGlowShell(parentMesh) {
  // Outer additive glow halo — slightly enlarged, renders AFTER brain (renderOrder=3)
  // BackSide + depthWrite:false → pure additive glow, never clips into brain geometry
  const mat = new THREE.MeshBasicMaterial({
    color:       0x5865f2,
    transparent: true,
    opacity:     0.038,
    side:        THREE.BackSide,
    depthWrite:  false,
    blending:    THREE.AdditiveBlending,
  })
  const shell = new THREE.Mesh(parentMesh.geometry.clone(), mat)
  shell.scale.setScalar(1.022)   // 2.2% larger — visible halo without overlap
  shell.renderOrder = 3
  parentMesh.add(shell)
  S.glowMesh = shell
}

function addStarField() {
  const n = 2800, pos = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    const r = 400 + Math.random() * 300
    const θ = Math.random() * Math.PI * 2
    const φ = Math.acos(2 * Math.random() - 1)
    pos[i*3]   = r * Math.sin(φ) * Math.cos(θ)
    pos[i*3+1] = r * Math.sin(φ) * Math.sin(θ)
    pos[i*3+2] = r * Math.cos(φ)
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  const mat = new THREE.PointsMaterial({ color: 0x8899dd, size: 0.65, sizeAttenuation: true, transparent: true, opacity: 0.28 })
  scene.add(new THREE.Points(geo, mat))
}

// ─────────────────────────────────────────────────────────────────────────────
// BOLD playback
// ─────────────────────────────────────────────────────────────────────────────
function seekToTR(tr) {
  const clamped = Math.max(0, Math.min(tr, Math.max(0, S.nTRs - 1)))
  S.prevTR = S.currentTR
  S.currentTR = clamped

  if (S.boldFrames[clamped]) {
    const prev = S.boldFrames[S.prevTR] ?? S.boldFrames[clamped]
    uploadBOLDFrame(S.brainMesh.geometry, S.boldFrames[clamped], prev)
    if (S.boldMat) S.boldMat.uniforms.u_blend.value = 1.0
  }
  refreshTimelineUI()
  refreshAnalyticsForTR(clamped)
  updateSegHighlight(clamped)
}

function tickPlayback(nowMs) {
  if (!S.playing || S.nTRs === 0) return
  const delta   = nowMs - (S._lastTick || nowMs)
  S._lastTick   = nowMs
  S._accumMs   += delta * S.playSpeed
  const msPerTR = TR_SECONDS * 1000
  while (S._accumMs >= msPerTR) {
    S._accumMs -= msPerTR
    seekToTR((S.currentTR + 1) % S.nTRs)
  }
  // Smooth blend: 0→1 within current TR window
  if (S.boldMat) {
    S.boldMat.uniforms.u_blend.value = Math.min(S._accumMs / msPerTR, 1.0)
  }
}

function play() {
  if (S.boldFrames.filter(Boolean).length === 0) return
  S.playing    = true
  S._accumMs   = 0
  S._lastTick  = performance.now()
  $play.textContent   = '⏸'
  controls.autoRotate = false
  gsap.to(S.boldMat?.uniforms.u_alpha_global ?? {}, { value: 0.92, duration: 0.4 })
}

function pause() {
  S.playing = false
  $play.textContent = '▶'
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline UI
// ─────────────────────────────────────────────────────────────────────────────
function refreshTimelineUI() {
  const t = S.currentTR * TR_SECONDS
  $tlTime.textContent  = t.toFixed(1) + ' s'
  $currentTR.textContent = `${S.currentTR}`

  if (!S._dragging && S.nTRs > 0) {
    const pct = (S.currentTR / Math.max(1, S.nTRs - 1)) * 100
    $slider.value = pct
    // Drive CSS gradient fill on slider
    $slider.style.setProperty('--pct', pct + '%')
  }
}

function buildTickMarks() {
  $tlTicks.innerHTML = ''
  if (S.nTRs === 0) return
  const nSegs = Math.ceil(S.nTRs / TRS_PER_SEG)
  for (let i = 0; i < nSegs; i++) {
    const div = document.createElement('div')
    div.className = 'tl-tick'
    div.dataset.seg = i
    div.title = `${i * 5}–${(i + 1) * 5} s`
    div.addEventListener('click', () => seekToTR(i * TRS_PER_SEG))
    $tlTicks.appendChild(div)
  }
}

function updateSegHighlight(tr) {
  const segIdx = Math.floor(tr / TRS_PER_SEG)
  document.querySelectorAll('.tl-tick').forEach((el, i) => {
    el.classList.toggle('active', i === segIdx)
  })
  document.querySelectorAll('.seg-row').forEach((el, i) => {
    el.classList.toggle('active', i === segIdx)
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Analytics panel
// ─────────────────────────────────────────────────────────────────────────────
function buildNetworkList() {
  $netList.innerHTML = YEO7_NAMES.map((name, i) => `
    <div class="network-row" data-net="${i}" title="Click to highlight ${name} network">
      <div class="net-swatch" style="background:${YEO7_HEX[i]}"></div>
      <div class="net-name">${name}</div>
      <div class="net-bar-bg">
        <div class="net-bar" id="nb-${i}" style="width:0%;background:${YEO7_HEX[i]}"></div>
      </div>
      <div class="net-z" id="nz-${i}">—</div>
    </div>
  `).join('')

  $netList.querySelectorAll('.network-row').forEach(row => {
    row.addEventListener('click', () => {
      const n = parseInt(row.dataset.net)
      flashNetworkHighlight(n)
    })
  })
}

function flashNetworkHighlight(netIdx) {
  if (!S.boldMat) return
  const prev = S.mode
  setMode(1)  // show network overlay
  setTimeout(() => setMode(prev), 2500)
}

function updateNetworkBars(yeo7Scores) {
  const vals  = YEO7_NAMES.map(n => yeo7Scores[n] ?? 0)
  const maxV  = Math.max(...vals, 0.001)
  vals.forEach((v, i) => {
    const pct = Math.min((v / maxV) * 100, 100)
    const bar = document.getElementById(`nb-${i}`)
    const zEl = document.getElementById(`nz-${i}`)
    if (bar) bar.style.width = pct + '%'
    if (zEl) zEl.textContent = v > 0 ? v.toFixed(2) : '—'
  })
}

function updateROIList(topRois) {
  if (!topRois || !topRois.length) {
    $roiList.innerHTML = '<div style="font-size:10px;color:var(--text-dim);font-family:var(--mono);">No ROI data</div>'
    return
  }
  $roiList.innerHTML = topRois.slice(0, 6).map((roi, i) => `
    <div class="roi-item">
      <span class="roi-rank">${i + 1}.</span>
      <span class="roi-name" title="${roi.name ?? roi}">${roi.name ?? roi}</span>
      <span class="roi-val">${roi.z_score !== undefined ? roi.z_score.toFixed(2) + 'σ' : ''}</span>
    </div>
  `).join('')
}

// Per-TR analytics (update peak time text if current TR is at peak)
function refreshAnalyticsForTR(tr) {
  const t = tr * TR_SECONDS
  if (S.analysis) {
    const pct = getActivePctAtTR(tr)
    if (pct !== null) {
      setText($cortexPct, pct.toFixed(1), '%')
    }
    updateROIList(S.analysis.top_rois)
  }
}

function getActivePctAtTR(tr) {
  if (!S.brainMesh || !S.boldFrames[tr] || !S.analysis) return null
  const frame     = S.boldFrames[tr]
  const threshold = S.boldMat?.uniforms.u_threshold.value ?? 0.5
  let   above     = 0
  for (let i = 0; i < frame.length; i++) if (Math.abs(frame[i]) > threshold) above++
  return (above / frame.length) * 100
}

// ─────────────────────────────────────────────────────────────────────────────
// 5-second segment analysis
// ─────────────────────────────────────────────────────────────────────────────
function computeSegments() {
  if (!S.boldFrames.length || !S.nTRs) return
  const nSegs = Math.ceil(S.nTRs / TRS_PER_SEG)
  S.segments  = []

  for (let s = 0; s < nSegs; s++) {
    const startTR = s * TRS_PER_SEG
    const endTR   = Math.min(startTR + TRS_PER_SEG, S.nTRs)
    let maxZ = 0, domNet = -1
    const netSums = new Float64Array(7)
    let   netCnts = 0

    for (let t = startTR; t < endTR; t++) {
      const frame = S.boldFrames[t]
      if (!frame) continue
      for (let v = 0; v < frame.length; v++) {
        const z = Math.abs(frame[v])
        if (z > maxZ) maxZ = z
        if (z > 0.5) {
          const net = getNetworkAtVert(v)
          if (net >= 0 && net < 7) { netSums[net] += z; netCnts++ }
        }
      }
    }

    // Dominant network in this segment
    let domIdx = -1, domVal = 0
    for (let n = 0; n < 7; n++) {
      if (netSums[n] > domVal) { domVal = netSums[n]; domIdx = n }
    }

    S.segments.push({
      startTR, endTR,
      startS: startTR * TR_SECONDS,
      endS:   endTR * TR_SECONDS,
      maxZ:   maxZ,
      domNet: domIdx >= 0 ? YEO7_NAMES[domIdx] : 'Unknown',
      domIdx,
    })
  }

  renderSegmentList()
  buildTickMarks()
}

function renderSegmentList() {
  if (!S.segments.length) return
  const maxZ = Math.max(...S.segments.map(s => s.maxZ), 0.001)

  $segList.innerHTML = S.segments.map((seg, i) => {
    const pct     = Math.round((seg.maxZ / maxZ) * 100)
    const barCol  = seg.domIdx >= 0 ? YEO7_HEX[seg.domIdx] : '#444'
    const timeStr = `${seg.startS.toFixed(0)}–${seg.endS.toFixed(0)}s`
    return `
      <div class="seg-row" data-seg="${i}" data-tr="${seg.startTR}" title="Jump to ${timeStr}">
        <div class="seg-dot" style="background:${barCol}"></div>
        <div class="seg-time">${timeStr}</div>
        <div class="seg-bar-wrap">
          <div class="seg-bar" style="width:${pct}%;background:${barCol}"></div>
        </div>
        <div class="seg-net">${seg.domNet}</div>
      </div>
    `
  }).join('')

  $segList.querySelectorAll('.seg-row').forEach(row => {
    row.addEventListener('click', () => {
      seekToTR(parseInt(row.dataset.tr))
      if (!S.playing) play()
    })
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Network activity sparkline
// ─────────────────────────────────────────────────────────────────────────────
function computeActivityHistory() {
  if (!S.boldFrames.length) return
  S.activityHistory = {}

  for (const net of YEO7_NAMES) {
    S.activityHistory[net] = new Float32Array(S.nTRs)
  }

  // Vertex → network lookup (build once)
  const netAttr = S.brainMesh?.geometry.getAttribute('network')
  if (!netAttr) return

  const vtxNets = new Int16Array(S.nVertices)
  for (let v = 0; v < S.nVertices; v++) vtxNets[v] = Math.round(netAttr.getX(v))

  // Count vertices per network
  const netVtxCount = new Float64Array(7)
  for (let v = 0; v < S.nVertices; v++) {
    const n = vtxNets[v]
    if (n >= 0 && n < 7) netVtxCount[n]++
  }

  for (let t = 0; t < S.nTRs; t++) {
    const frame = S.boldFrames[t]
    if (!frame) continue
    const netSum = new Float64Array(7)
    for (let v = 0; v < S.nVertices; v++) {
      const n = vtxNets[v]
      if (n >= 0 && n < 7) netSum[n] += Math.abs(frame[v])
    }
    for (let n = 0; n < 7; n++) {
      const name = YEO7_NAMES[n]
      S.activityHistory[name][t] = netVtxCount[n] > 0
        ? netSum[n] / netVtxCount[n]
        : 0
    }
  }

  drawActivityChart()
}

function drawActivityChart() {
  const canvas = $actCanvas
  if (!canvas || !S.nTRs) return

  const W = canvas.offsetWidth || 220
  const H = canvas.offsetHeight || 56
  canvas.width  = W * window.devicePixelRatio
  canvas.height = H * window.devicePixelRatio

  const ctx = canvas.getContext('2d')
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
  ctx.clearRect(0, 0, W, H)

  // Background
  ctx.fillStyle = 'rgba(0,0,0,0.3)'
  ctx.fillRect(0, 0, W, H)

  // Global max for normalisation
  let gMax = 0.001
  for (const hist of Object.values(S.activityHistory)) {
    for (const v of hist) if (v > gMax) gMax = v
  }

  YEO7_NAMES.forEach((name, ni) => {
    const hist = S.activityHistory[name]
    if (!hist) return
    const hex = YEO7_HEX[ni]
    ctx.beginPath()
    ctx.strokeStyle = hex
    ctx.lineWidth   = 1.3
    ctx.globalAlpha = 0.75

    for (let t = 0; t < S.nTRs; t++) {
      const x = (t / (S.nTRs - 1)) * W
      const y = H - (hist[t] / gMax) * (H - 4)
      if (t === 0) ctx.moveTo(x, y)
      else         ctx.lineTo(x, y)
    }
    ctx.stroke()
  })
  ctx.globalAlpha = 1

  // Playhead
  if (S.nTRs > 0) {
    const px = (S.currentTR / (S.nTRs - 1)) * W
    ctx.strokeStyle = 'rgba(255,255,255,0.5)'
    ctx.lineWidth   = 1
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke()
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HRF curve
// ─────────────────────────────────────────────────────────────────────────────
function drawHRFCurve(peakTR = null) {
  const canvas = $hrfCanvas
  if (!canvas) return

  const W = canvas.offsetWidth || 180
  const H = 48
  canvas.width  = W * window.devicePixelRatio
  canvas.height = H * window.devicePixelRatio
  const ctx = canvas.getContext('2d')
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
  ctx.clearRect(0, 0, W, H)

  // Double-gamma HRF (canonical SPM model, sampled at 0.5s TR)
  const nSamples = 40
  const hrf = []
  for (let i = 0; i < nSamples; i++) {
    const t = i * TR_SECONDS
    const a1 = 6, a2 = 16, b1 = 1, b2 = 1, c = 1/6
    const g1 = Math.pow(t, a1-1) * Math.exp(-t/b1) / (Math.pow(b1,a1) * gamma(a1))
    const g2 = Math.pow(t, a2-1) * Math.exp(-t/b2) / (Math.pow(b2,a2) * gamma(a2))
    hrf.push(g1 - c * g2)
  }
  const mx = Math.max(...hrf, 0.001)
  const mn = Math.min(...hrf, 0)
  const rng = mx - mn

  // Background
  ctx.fillStyle = 'rgba(0,0,0,0.25)'
  ctx.roundRect(0, 0, W, H, 3)
  ctx.fill()

  // Axes
  const yZero = H - ((0 - mn) / rng) * (H - 6) - 3
  ctx.strokeStyle = 'rgba(255,255,255,0.1)'
  ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(0, yZero); ctx.lineTo(W, yZero); ctx.stroke()

  // HRF curve
  const grad = ctx.createLinearGradient(0, 0, W, 0)
  grad.addColorStop(0, '#5865f2')
  grad.addColorStop(1, '#eb459e')
  ctx.beginPath()
  ctx.strokeStyle = grad
  ctx.lineWidth   = 2

  for (let i = 0; i < nSamples; i++) {
    const x = (i / (nSamples - 1)) * W
    const y = H - ((hrf[i] - mn) / rng) * (H - 6) - 3
    if (i === 0) ctx.moveTo(x, y)
    else         ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Peak marker
  if (peakTR !== null && S.nTRs > 0) {
    const px = Math.min(peakTR, nSamples - 1)
    const x  = (px / (nSamples - 1)) * W
    ctx.strokeStyle = '#57f287'
    ctx.lineWidth   = 1.5
    ctx.setLineDash([2, 2])
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
    ctx.setLineDash([])
  }
}

// Stirling approximation for gamma function
function gamma(n) {
  if (n <= 1) return 1
  return (n - 1) * gamma(n - 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Narration panel
// ─────────────────────────────────────────────────────────────────────────────
function renderNarration(tier) {
  const text = S.narrations[tier]
  if (!text) {
    $narration.innerHTML = `<p style="color:var(--text-dim);font-style:italic;font-size:11px;">
      Tier ${tier} narration will appear after analysis…</p>`
    return
  }
  const html = text
    .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
  $narration.innerHTML = `<p>${html}</p>`
  gsap.fromTo($narration, { opacity: 0 }, { opacity: 1, duration: 0.35, ease: 'power2.out' })
}

function flashNarration(html) {
  const saved = $narration.innerHTML
  $narration.innerHTML = html
  gsap.fromTo($narration, { opacity: 0 }, { opacity: 1, duration: 0.25 })
  setTimeout(() => {
    $narration.innerHTML = saved
    gsap.fromTo($narration, { opacity: 0 }, { opacity: 1, duration: 0.25 })
  }, 4500)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D CSS Labels (region names anchored to brain)
// ─────────────────────────────────────────────────────────────────────────────
function buildBrainLabels(regionLabels = REGION_LABELS_DEFAULT) {
  // Remove existing
  S.labelEls.forEach(el => el.remove())
  S.labelEls = []

  regionLabels.forEach(({ name, pos }) => {
    const el = document.createElement('div')
    el.className = 'brain-label'
    el.textContent = name
    $labelCont.appendChild(el)
    S.labelEls.push({ el, worldPos: pos.clone() })
  })
}

function updateLabelPositions() {
  if (!S.labelsVisible) return
  const w = window.innerWidth, h = window.innerHeight
  S.labelEls.forEach(({ el, worldPos }) => {
    const v = worldPos.clone().project(camera)
    const x = (v.x * 0.5 + 0.5) * w
    const y = (-(v.y) * 0.5 + 0.5) * h
    const behind = v.z > 1

    el.style.left    = x + 'px'
    el.style.top     = y + 'px'
    el.style.display = behind ? 'none' : 'block'
    el.style.opacity = behind ? '0' : String(Math.max(0, Math.min(1, 1 - (v.z - 0.5) * 2)))
  })
}

function setLabelsVisible(vis) {
  S.labelsVisible = vis
  S.labelEls.forEach(({ el }) => {
    el.style.display = vis ? 'block' : 'none'
  })
  $btnLabels.classList.toggle('active', vis)
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour legend
// ─────────────────────────────────────────────────────────────────────────────
function updateLegend(mapIdx) {
  const name = COLOURMAP_NAMES[mapIdx] ?? 'inferno'
  $legendBar.style.background   = buildLegendGradient(name)
  $legendMapName.textContent    = name
  if (S.boldMat) {
    const lo = S.boldMat.uniforms.u_threshold.value.toFixed(1)
    const hi = S.boldMat.uniforms.u_max.value.toFixed(1)
    $legendLo.textContent = lo + 'σ'
    $legendHi.textContent = hi + 'σ'
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode switching
// ─────────────────────────────────────────────────────────────────────────────
function setMode(mode) {
  S.mode = mode
  if (S.boldMat) S.boldMat.uniforms.u_mode.value = mode
  $btnBold.classList.toggle('active', mode === 0)
  $btnNet.classList.toggle('active',  mode === 1)
  $btnHemi.classList.toggle('active', mode === 2)
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket
// ─────────────────────────────────────────────────────────────────────────────
function connectWS() {
  try { S.ws = new WebSocket(WS_URL) }
  catch { setTimeout(connectWS, 3000); return }

  S.ws.binaryType = 'arraybuffer'

  S.ws.onopen  = () => {
    setStatusDot('#57f287')
    console.log('[WS] connected')
  }
  S.ws.onclose = () => {
    setStatusDot('#ed4245')
    setTimeout(connectWS, 3000)
  }
  S.ws.onerror = () => setStatusDot('#faa61a')

  S.ws.onmessage = ev => {
    if (ev.data instanceof ArrayBuffer) {
      handleBinaryFrame(ev.data)
    } else {
      try { handleJSON(JSON.parse(ev.data)) }
      catch (e) { console.warn('[WS] bad JSON', e) }
    }
  }
}

function setStatusDot(color) {
  const dot = document.getElementById('status-dot')
  if (dot) { dot.style.background = color; dot.style.boxShadow = `0 0 8px ${color}` }
}

function handleBinaryFrame(buf) {
  // Protocol: uint32 frame_idx (4 bytes LE) + float32[n_verts]
  const view    = new DataView(buf)
  const frameIdx = view.getUint32(0, true)
  const bold     = new Float32Array(buf, 4)
  storeBoldFrame(frameIdx, bold)
}

function handleJSON(msg) {
  switch (msg.type) {
    case 'session_init':  onSessionInit(msg);  break
    case 'bold_all':      onBoldAll(msg);       break
    case 'bold_frame':    onBoldFrameJSON(msg); break
    case 'narration':     onNarration(msg);     break
    case 'analysis':      onAnalysis(msg);      break
    case 'stimulus':      onStimulus(msg);      break
    case 'ping':          break
    default:              console.debug('[WS]', msg.type)
  }
}

function onSessionInit(msg) {
  S.nTRs      = msg.n_trs ?? 0
  S.boldFrames = new Array(S.nTRs).fill(null)
  S.currentTR  = 0
  S._accumMs   = 0
  if (msg.stimulus_title) $stimTitle.textContent = msg.stimulus_title
  if (msg.n_trs) $stimMeta.textContent = `${msg.n_trs} TRs · ${(msg.n_trs * TR_SECONDS).toFixed(1)}s · 2 Hz`
  $slider.max = Math.max(100, S.nTRs)
  buildTickMarks()
}

function onBoldAll(msg) {
  S.nTRs       = msg.n_trs
  S.boldFrames = msg.bold_data.map(row => Float32Array.from(row))
  $slider.max  = S.nTRs
  buildTickMarks()
  computeSegments()
  computeActivityHistory()
  drawHRFCurve(S.analysis?.peak_t ? S.analysis.peak_t / TR_SECONDS : null)
  seekToTR(0)
  play()
}

function onBoldFrameJSON(msg) {
  storeBoldFrame(msg.frame_idx, Float32Array.from(msg.bold_zscore))
}

function storeBoldFrame(idx, bold) {
  if (idx >= S.boldFrames.length) S.boldFrames.length = idx + 1
  S.boldFrames[idx] = bold
  if (idx === S.currentTR && S.brainMesh) {
    uploadBOLDFrame(S.brainMesh.geometry, bold)
  }
  // When all frames received, post-process
  const received = S.boldFrames.filter(Boolean).length
  if (received === S.nTRs && S.nTRs > 0) {
    computeSegments()
    computeActivityHistory()
    drawHRFCurve(S.analysis?.peak_t ? S.analysis.peak_t / TR_SECONDS : null)
    if (!S.playing) play()
  }
}

function onNarration(msg) {
  S.narrations[msg.tier] = msg.text
  if (msg.tier === S.activeTier) renderNarration(msg.tier)
}

function onAnalysis(msg) {
  S.analysis = msg

  // Peak time
  if (msg.peak_t !== undefined) {
    const peakEl = document.querySelector('#peak-time')
    if (peakEl) {
      peakEl.innerHTML = `${msg.peak_t.toFixed(1)}<span class="metric-unit">s</span>`
    }
    $peakRegion.textContent = msg.top_rois?.[0]?.name ?? msg.top_rois?.[0] ?? '—'
  }
  if (msg.dominant_network) {
    $domNet.textContent = msg.dominant_network
    $domSub.textContent = `${msg.cortex_pct?.toFixed(1) ?? '—'}% cortex active`
  }
  if (msg.cortex_pct !== undefined) {
    $cortexPct.innerHTML = `${msg.cortex_pct.toFixed(1)}<span class="metric-unit">%</span>`
  }
  if (msg.yeo7_scores)  updateNetworkBars(msg.yeo7_scores)
  if (msg.top_rois)     updateROIList(msg.top_rois)
  if (msg.peak_t)       drawHRFCurve(msg.peak_t / TR_SECONDS)

  // Set HRF peak uniform
  if (S.boldMat && msg.peak_t && S.nTRs) {
    S.boldMat.uniforms.u_hrf_peak.value = (msg.peak_t / TR_SECONDS) / S.nTRs
  }
}

function onStimulus(msg) {
  if (msg.title) $stimTitle.textContent = msg.title
  if (msg.meta)  $stimMeta.textContent  = msg.meta
}

// ─────────────────────────────────────────────────────────────────────────────
// Controls
// ─────────────────────────────────────────────────────────────────────────────
$play.addEventListener('click', () => S.playing ? pause() : play())

$slider.addEventListener('mousedown', () => { S._dragging = true; pause() })
$slider.addEventListener('touchstart', () => { S._dragging = true; pause() })
$slider.addEventListener('input', () => {
  if (!S.nTRs) return
  const tr = Math.round((+$slider.value / +$slider.max) * (S.nTRs - 1))
  seekToTR(tr)
  $slider.style.setProperty('--pct', $slider.value + '%')
})
$slider.addEventListener('mouseup',  () => { S._dragging = false; play() })
$slider.addEventListener('touchend', () => { S._dragging = false; play() })

$speedBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    $speedBtns.forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    S.playSpeed = parseFloat(btn.dataset.speed)
    S._accumMs  = 0
  })
})

$tierBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    $tierBtns.forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    S.activeTier = parseInt(btn.dataset.tier)
    renderNarration(S.activeTier)
  })
})

$btnBold.addEventListener('click',  () => setMode(0))
$btnNet.addEventListener('click',   () => setMode(1))
$btnHemi.addEventListener('click',  () => setMode(2))
$btnLabels.addEventListener('click',() => setLabelsVisible(!S.labelsVisible))
$btnSpin.addEventListener('click',  () => {
  controls.autoRotate = !controls.autoRotate
  $btnSpin.classList.toggle('active', controls.autoRotate)
})

// ─────────────────────────────────────────────────────────────────────────────
// Audience mode switcher
// Three levels: Student (0) · Public (1) · Expert (2)
// Affects: label text, UI complexity, narration tier pre-select,
//          network name vocabulary, and visible stats panels.
// ─────────────────────────────────────────────────────────────────────────────
const AUDIENCE_CONFIGS = [
  {
    id:          'student',
    label:       '🎓 Student',
    tier:        3,   // High school narration
    netNames: {
      Vis: 'Vision 👁', SomMot: 'Movement & Touch 🖐',
      DorsAttn: 'Focus & Attention 🎯', SalVentAttn: 'Alertness ⚡',
      Limbic: 'Emotions & Memory 💭', Cont: 'Thinking & Planning 🧩',
      Default: 'Daydreaming 🌙',
    },
    hideStats:   true,   // hide peak z, laterality numbers
    description: 'Brain regions explained in plain language',
    threshold:   0.7,    // slightly higher threshold → cleaner, less noise
  },
  {
    id:          'public',
    label:       '👥 Public',
    tier:        2,   // Curious adult narration
    netNames: {
      Vis: 'Visual cortex', SomMot: 'Motor cortex',
      DorsAttn: 'Attention network', SalVentAttn: 'Salience network',
      Limbic: 'Limbic system', Cont: 'Control network',
      Default: 'Default mode network',
    },
    hideStats:   false,
    description: 'What your brain is doing, in everyday terms',
    threshold:   0.5,
  },
  {
    id:          'expert',
    label:       '🩺 Expert',
    tier:        6,   // Researcher narration
    netNames: {
      Vis: 'Visual (Yeo-Vis)', SomMot: 'Somatomotor',
      DorsAttn: 'Dorsal Attention', SalVentAttn: 'Salience/Ventral Attn',
      Limbic: 'Limbic', Cont: 'Frontoparietal Control',
      Default: 'Default Mode',
    },
    hideStats:   false,
    description: 'Full clinical detail — Yeo-7, ROI z-scores, laterality',
    threshold:   0.3,   // lower threshold → see subtle activations
  },
]

let _audienceMode = 1  // default: Public

function setAudienceMode(idx) {
  _audienceMode = Math.max(0, Math.min(idx, AUDIENCE_CONFIGS.length - 1))
  const cfg = AUDIENCE_CONFIGS[_audienceMode]

  // Update narration tier pre-selection
  S.activeTier = cfg.tier
  renderNarration(cfg.tier)

  // Update threshold (expert sees more subtle activation)
  if (S.boldMat) S.boldMat.uniforms.u_threshold.value = cfg.threshold

  // Update audience buttons
  document.querySelectorAll('.audience-btn').forEach((b, i) => {
    b.classList.toggle('active', i === _audienceMode)
  })

  // Update Yeo-7 network names in sidebar
  document.querySelectorAll('[data-net-name]').forEach(el => {
    const key = el.dataset.netName
    if (cfg.netNames[key]) el.textContent = cfg.netNames[key]
  })

  // Toggle technical stats panels
  const statsEls = document.querySelectorAll('[data-expert-only]')
  statsEls.forEach(el => {
    el.style.display = cfg.hideStats ? 'none' : ''
  })

  // Update page subtitle / description
  const $desc = document.getElementById('audience-desc')
  if ($desc) $desc.textContent = cfg.description
}

// Inject audience mode buttons into DOM
;(function injectAudiencePicker() {
  const container = document.createElement('div')
  container.id = 'audience-picker'
  container.style.cssText = [
    'position:fixed', 'top:10px', 'left:50%', 'transform:translateX(-50%)',
    'display:flex', 'gap:6px', 'z-index:300', 'pointer-events:all',
  ].join(';')

  AUDIENCE_CONFIGS.forEach((cfg, i) => {
    const btn = document.createElement('button')
    btn.className   = 'audience-btn tl-speed' + (i === _audienceMode ? ' active' : '')
    btn.textContent = cfg.label
    btn.title       = cfg.description
    btn.style.cssText = 'font-size:12px;padding:4px 10px;border-radius:16px;'
    btn.addEventListener('click', () => setAudienceMode(i))
    container.appendChild(btn)
  })

  // Description subtitle
  const desc = document.createElement('div')
  desc.id = 'audience-desc'
  desc.style.cssText = [
    'position:fixed', 'top:42px', 'left:50%', 'transform:translateX(-50%)',
    'font-size:11px', 'color:rgba(255,255,255,0.5)', 'z-index:300',
    'pointer-events:none', 'white-space:nowrap',
  ].join(';')
  desc.textContent = AUDIENCE_CONFIGS[_audienceMode].description

  document.body.appendChild(container)
  document.body.appendChild(desc)
})()

// Colour map picker — inject into timeline
;(function injectColourPicker() {
  const tl   = document.getElementById('timeline')
  if (!tl) return
  const wrap = document.createElement('div')
  wrap.style.cssText = 'display:flex;gap:3px;flex-shrink:0;'
  const labels = { inferno:'🔥', plasma:'💜', turbo:'🌈', coolwarm:'🌡', viridis:'🌿', rdbu:'±' }
  COLOURMAP_NAMES.forEach((name, i) => {
    const b = document.createElement('button')
    b.className   = 'tl-speed' + (i === 0 ? ' active' : '')
    b.textContent = labels[name] ?? name
    b.title       = name
    b.addEventListener('click', () => {
      wrap.querySelectorAll('button').forEach(x => x.classList.remove('active'))
      b.classList.add('active')
      S.colourMapIdx = i
      if (S.boldMat) S.boldMat.uniforms.u_colormap.value = i
      updateLegend(i)
    })
    wrap.appendChild(b)
  })
  tl.appendChild(wrap)
})()

// Keyboard shortcuts
window.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT') return
  switch (ev.code) {
    case 'Space':       ev.preventDefault(); S.playing ? pause() : play(); break
    case 'ArrowRight':  seekToTR(S.currentTR + 1); break
    case 'ArrowLeft':   seekToTR(S.currentTR - 1); break
    case 'KeyB':        setMode(0); break
    case 'KeyN':        setMode(1); break
    case 'KeyH':        setMode(2); break
    case 'KeyL':        setLabelsVisible(!S.labelsVisible); break
    case 'KeyR':        controls.autoRotate = !controls.autoRotate; break
    // PBR tweaks (Q/W = roughness ↓/↑, E/F = SSS ↓/↑)
    case 'KeyQ':
      _pbrRoughness = Math.max(0.05, _pbrRoughness - 0.05)
      _updatePBR()
      break
    case 'KeyW':
      _pbrRoughness = Math.min(1.0, _pbrRoughness + 0.05)
      _updatePBR()
      break
    case 'KeyE':
      _pbrSSS = Math.max(0.0, _pbrSSS - 0.05)
      _updatePBR()
      break
    case 'KeyF':
      _pbrSSS = Math.min(1.0, _pbrSSS + 0.05)
      _updatePBR()
      break
  }
})

// Drag-and-drop
document.body.addEventListener('dragover', ev => {
  ev.preventDefault()
  document.body.classList.add('drag-over')
})
document.body.addEventListener('dragleave', () => document.body.classList.remove('drag-over'))
document.body.addEventListener('drop', async ev => {
  ev.preventDefault()
  document.body.classList.remove('drag-over')
  const file = ev.dataTransfer?.files?.[0]
  if (!file) return
  await uploadMedia(file)
})

async function uploadMedia(file) {
  const ok = file.type.startsWith('video/') || file.type.startsWith('audio/') ||
             /\.(mp4|mov|mkv|webm|avi|mp3|wav|m4a|flac)$/i.test(file.name)
  if (!ok) { alert('Please drop a video or audio file.'); return }

  $stimTitle.textContent = file.name
  $stimMeta.textContent  = `Uploading ${(file.size / 1e6).toFixed(1)} MB…`
  showLoading('Uploading to TRIBE v2 pipeline…')

  const form = new FormData()
  form.append('file', file)
  try {
    const r  = await fetch('/api/submit', { method: 'POST', body: form })
    const d  = await r.json()
    hideLoading()
    $stimMeta.textContent = `Job ${d.job_id} — running pipeline…`
  } catch (e) {
    hideLoading()
    $stimMeta.textContent = `Upload failed: ${e.message}`
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading overlay
// ─────────────────────────────────────────────────────────────────────────────
function showLoading(msg = 'Loading…') {
  $loadText.textContent = msg
  $loading.classList.remove('hidden')
  $loading.style.opacity = '1'
}
function hideLoading() {
  gsap.to($loading, { opacity: 0, duration: 0.6, onComplete: () => {
    $loading.classList.add('hidden')
    $loading.style.opacity = ''
  }})
}
function setStage(msg, pct) {
  $loadStage.textContent = msg
  if ($loadBar && pct !== undefined) $loadBar.style.width = pct + '%'
}
function setProgress(pct) {
  if ($loadBar) $loadBar.style.width = Math.round(pct) + '%'
}
function setText(el, val, unit = '') {
  if (!el) return
  el.innerHTML = `${val}<span class="metric-unit">${unit}</span>`
}

// ─────────────────────────────────────────────────────────────────────────────
// Window resize
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
  drawActivityChart()
  drawHRFCurve()
})

// ─────────────────────────────────────────────────────────────────────────────
// Render loop
// ─────────────────────────────────────────────────────────────────────────────
let _raf = 0
const clock = new THREE.Clock()

function animate(nowMs) {
  _raf = requestAnimationFrame(animate)
  stats.begin()

  const elapsed = clock.getElapsedTime()
  if (S.boldMat) S.boldMat.uniforms.u_time.value = elapsed

  controls.update()
  tickPlayback(nowMs)
  doRaycast()
  updateLabelPositions()

  // Redraw sparkline playhead (cheap, canvas 2D)
  if (S.nTRs > 0 && S.playing) {
    const w  = $actCanvas.offsetWidth || 220
    const px = (S.currentTR / (S.nTRs - 1)) * w
    // Only redraw if moved more than 2px
    if (Math.abs(px - (_raf % 999)) > 2) drawActivityChart()
  }

  renderer.render(scene, camera)
  stats.end()
}

// ─────────────────────────────────────────────────────────────────────────────
// URL param result loading
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Parse ?r=<job_id> from the URL. If present, load that specific result.
 * Otherwise fall back to /api/result/latest.
 * Supports both JSON endpoint (small results) and binary bold endpoint (large).
 */
async function loadResultFromURL() {
  const params  = new URLSearchParams(location.search)
  const jobId   = params.get('r') || params.get('result')
  const endpoint = jobId ? `/api/result/${jobId}` : '/api/result/latest'

  setStage('Loading brain activity data…')

  try {
    const r = await fetch(endpoint)
    if (!r.ok) {
      if (!jobId) return   // no cached result, just wait for WS
      throw new Error(`Result "${jobId}" not found (HTTP ${r.status})`)
    }
    const d = await r.json()

    // Populate stimulus panel
    if (d.stimulus_title || d.media_filename) {
      $stimTitle.textContent = d.stimulus_title || d.media_filename || '—'
    }
    if (d.n_trs) {
      $stimMeta.textContent = `${d.n_trs} TRs · ${(d.n_trs * TR_SECONDS).toFixed(1)}s · 2 Hz`
      if (jobId) $stimMeta.textContent += ` · job: ${jobId}`
    }

    // Narrations
    if (d.narrations) {
      S.narrations = d.narrations
      renderNarration(S.activeTier)
    }

    // Analysis
    if (d.analysis) onAnalysis(d.analysis)

    // BOLD data — try binary endpoint first (faster), fall back to JSON list
    if (jobId) {
      await loadBoldBinary(jobId, d.n_trs, d.n_verts ?? 20484)
    } else if (d.bold_data) {
      onBoldAll({ type: 'bold_all', n_trs: d.n_trs, bold_data: d.bold_data })
    }

    // Update page title
    if (d.stimulus_title) document.title = `JemmaBrain — ${d.stimulus_title}`

  } catch (e) {
    console.error('[JemmaBrain] result load failed:', e)
    if (jobId) setStage(`Could not load result "${jobId}": ${e.message}`)
  }
}

/**
 * Load BOLD data as binary from /api/result/{job_id}/bold
 * Binary protocol: 8-byte header (uint32 n_trs, uint32 n_verts) + float32 data
 */
async function loadBoldBinary(jobId, nTRs, nVerts) {
  try {
    const r = await fetch(`/api/result/${jobId}/bold`)
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const buf    = await r.arrayBuffer()
    const header = new DataView(buf, 0, 8)
    const n_trs  = header.getUint32(0, true)
    const n_verts= header.getUint32(4, true)
    const data   = new Float32Array(buf, 8)

    S.nTRs = n_trs
    S.boldFrames = []
    for (let t = 0; t < n_trs; t++) {
      S.boldFrames.push(data.subarray(t * n_verts, (t + 1) * n_verts).slice())
    }
    $slider.max = n_trs
    buildTickMarks()
    computeSegments()
    computeActivityHistory()
    drawHRFCurve(S.analysis?.peak_t ? S.analysis.peak_t / TR_SECONDS : null)
    seekToTR(0)
    play()
  } catch (e) {
    // Binary endpoint failed — try fetching JSON bold_data from meta
    console.warn('[JemmaBrain] binary bold failed, trying JSON fallback:', e)
    try {
      const r2 = await fetch(`/api/result/${jobId}`)
      if (r2.ok) {
        const d2 = await r2.json()
        if (d2.bold_data) onBoldAll({ type: 'bold_all', n_trs: d2.n_trs, bold_data: d2.bold_data })
      }
    } catch {}
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result gallery (shown when no ?r= param and no latest result)
// ─────────────────────────────────────────────────────────────────────────────

async function maybeShowGallery() {
  try {
    const r = await fetch('/api/results')
    if (!r.ok) return
    const results = await r.json()
    if (!results.length) return

    // Build a quick gallery overlay
    const overlay = document.createElement('div')
    overlay.id = 'result-gallery'
    overlay.style.cssText = `
      position:fixed; inset:0; z-index:150; background:rgba(7,11,18,0.92);
      backdrop-filter:blur(12px); display:flex; align-items:center; justify-content:center;
      flex-direction:column; gap:12px; padding:20px;
    `
    overlay.innerHTML = `
      <div style="font-size:16px;font-weight:600;margin-bottom:8px;">Past analyses — click to replay in 3D</div>
      <div id="gallery-list" style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;max-width:800px;overflow-y:auto;max-height:70vh;"></div>
      <button onclick="this.parentElement.remove()" style="margin-top:12px;padding:8px 20px;border-radius:8px;background:rgba(88,101,242,0.3);border:1px solid rgba(88,101,242,0.5);color:#fff;cursor:pointer;font-family:var(--mono);font-size:12px;">
        Skip — wait for new analysis
      </button>
    `
    const list = overlay.querySelector('#gallery-list')
    results.slice(0, 20).forEach(res => {
      const dt = new Date(res.timestamp * 1000)
      const card = document.createElement('a')
      card.href  = `/?r=${res.job_id}`
      card.style.cssText = `
        display:block;padding:12px 16px;background:rgba(10,15,26,0.85);
        border:1px solid rgba(88,101,242,0.25);border-radius:10px;
        text-decoration:none;color:inherit;min-width:200px;max-width:220px;
        transition:border-color 0.2s;cursor:pointer;
      `
      card.onmouseenter = () => card.style.borderColor = 'rgba(88,101,242,0.7)'
      card.onmouseleave = () => card.style.borderColor = 'rgba(88,101,242,0.25)'
      card.innerHTML = `
        <div style="font-size:12px;font-weight:500;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${res.stimulus_title}">${res.stimulus_title || res.media_filename || 'Unknown'}</div>
        <div style="font-size:9px;font-family:var(--mono);color:#94a3b8;">${dt.toLocaleDateString()} ${dt.toLocaleTimeString()}</div>
        <div style="font-size:9px;font-family:var(--mono);color:#94a3b8;margin-top:2px;">${res.n_trs} TRs · ${(res.n_trs * 0.5).toFixed(0)}s</div>
      `
      list.appendChild(card)
    })
    document.body.appendChild(overlay)
  } catch {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
  // ── Red Team Kitchen intro (once per session) ─────────────────────────────
  // Show first — before loading anything — so the user sees the recipe card
  // while the browser is idle. The card dismissal kicks off the brain load.
  await showRecipeIntro()

  showLoading('Initializing JemmaBrain · Red Team Kitchen')
  buildNetworkList()
  updateLegend(0)
  setMode(0)

  try {
    await loadBrainMesh()
    buildBrainLabels()
    drawHRFCurve()
  } catch (e) {
    console.error('[JemmaBrain] mesh load failed:', e)
    setStage(`Mesh error: ${e.message} — retrying in 5s`)
    setTimeout(main, 5000)
    return
  }

  hideLoading()

  // Load result from URL param ?r=<id> or latest
  await loadResultFromURL()

  // If still no BOLD data, show gallery of past results
  if (!S.boldFrames.filter(Boolean).length) {
    await maybeShowGallery()
  }

  connectWS()
  animate(performance.now())
}

main()
