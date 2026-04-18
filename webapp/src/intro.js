/**
 * intro.js — Red Team Kitchen recipe card intro overlay
 *
 * Shows a recipe card "segment intro" on first visit (once per session).
 * Returns a Promise that resolves when the user clicks "Start Cooking" or
 * if the session already saw the intro (skip it immediately).
 *
 * Usage in main.js:
 *   import { showRecipeIntro } from './intro.js'
 *   await showRecipeIntro()
 *   // ... proceed to load brain
 */

const RECIPE_DATA = {
  number:      '001',
  title:       'Predicted Neural Soufflé',
  emoji:       '🧠',
  prepTime:    '30 seconds',
  cookTime:    '4 – 7 minutes',
  serves:      '3 audience tiers',
  ingredients: [
    { emoji: '🎬', text: 'Any short video, audio, or text clip' },
    { emoji: '🧬', text: 'TRIBE v2 fMRI prediction model (CC-BY-NC 4.0)' },
    { emoji: '🤖', text: 'Gemma 4 (three sizes: E4B · 26B MoE · 31B)' },
    { emoji: '⚡', text: 'RTX 5090 Blackwell GPU (the oven)' },
    { emoji: '🎓', text: 'ISU neuroscience + psychology expertise' },
  ],
  method: [
    'Drop a video clip into the upload zone',
    'TRIBE v2 predicts which brain regions activate',
    'Gemma 4 narrates the result for your audience level',
    'Serve hot on the 3D cortical surface below',
  ],
  pairing:     'Best enjoyed with: curiosity, a neuroscience textbook, or a clinician',
  chef:        'Alexios Bluff Mara LLC',
  debut:       'Illinois State University · Spring 2026',
  season:      'Season 1, Episode 1',
}

// ── Styles (injected once) ────────────────────────────────────────────────────

function injectStyles() {
  if (document.getElementById('rtk-intro-styles')) return
  const style = document.createElement('style')
  style.id = 'rtk-intro-styles'
  style.textContent = /* css */`
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Crimson+Text:ital@0;1&family=Special+Elite&display=swap');

    :root {
      --rtk-red:    #c41e3a;
      --rtk-amber:  #d97706;
      --rtk-dark:   #0e0604;
      --rtk-cream:  #fef9f0;
      --rtk-sepia:  #7c6146;
      --rtk-rule:   #d4b896;
    }

    /* ── Overlay ──────────────────────────────────────────────────────────── */
    #rtk-intro {
      position: fixed;
      inset: 0;
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      background: radial-gradient(ellipse at center, #1a0a04 0%, #0a0400 70%);
      animation: rtk-fade-in 0.6s ease both;
    }
    @keyframes rtk-fade-in {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
    #rtk-intro.hiding {
      animation: rtk-fade-out 0.55s ease both;
    }
    @keyframes rtk-fade-out {
      from { opacity: 1; transform: scale(1); }
      to   { opacity: 0; transform: scale(1.03); }
    }

    /* ── Background steam particles ────────────────────────────────────────── */
    .rtk-steam {
      position: absolute;
      width: 3px;
      border-radius: 50%;
      background: rgba(255,220,180,0.08);
      animation: rtk-rise linear infinite;
      pointer-events: none;
    }
    @keyframes rtk-rise {
      0%   { transform: translateY(0) scaleX(1);  opacity: 0; }
      20%  { opacity: 1; }
      80%  { opacity: 0.4; }
      100% { transform: translateY(-320px) scaleX(2.5); opacity: 0; }
    }

    /* ── Recipe card ───────────────────────────────────────────────────────── */
    #rtk-card {
      position: relative;
      width: min(560px, 94vw);
      background: var(--rtk-cream);
      border-radius: 3px;
      box-shadow:
        0 2px 0 #b5996a,
        0 4px 0 #9e8558,
        0 6px 0 #887047,
        0 8px 0 #725d38,
        0 32px 80px rgba(0,0,0,0.8),
        0 0 0 1px rgba(0,0,0,0.3);
      overflow: hidden;
      animation: rtk-card-in 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.15s both;
    }
    @keyframes rtk-card-in {
      from { transform: translateY(40px) rotate(-1deg); opacity: 0; }
      to   { transform: translateY(0) rotate(0deg);    opacity: 1; }
    }

    /* ── Card header (red band) ────────────────────────────────────────────── */
    #rtk-header {
      background: var(--rtk-red);
      padding: 18px 24px 14px;
      text-align: center;
      position: relative;
      overflow: hidden;
    }
    #rtk-header::before {
      content: '';
      position: absolute;
      inset: 0;
      background: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 10px,
        rgba(0,0,0,0.06) 10px,
        rgba(0,0,0,0.06) 11px
      );
    }
    #rtk-wordmark {
      font-family: 'Special Elite', monospace;
      font-size: 22px;
      color: #fff;
      letter-spacing: 3px;
      text-transform: uppercase;
      position: relative;
    }
    #rtk-wordmark span {
      color: rgba(255,255,255,0.55);
      font-size: 11px;
      display: block;
      letter-spacing: 5px;
      margin-top: 2px;
      font-family: 'Crimson Text', serif;
    }
    #rtk-collab {
      font-family: 'Crimson Text', serif;
      font-style: italic;
      font-size: 12px;
      color: rgba(255,255,255,0.72);
      margin-top: 6px;
      position: relative;
    }
    #rtk-episode {
      position: absolute;
      top: 10px;
      right: 14px;
      font-family: 'Special Elite', monospace;
      font-size: 9px;
      color: rgba(255,255,255,0.45);
      letter-spacing: 1px;
      text-align: right;
      line-height: 1.5;
    }

    /* ── Recipe number banner ──────────────────────────────────────────────── */
    #rtk-recipe-num {
      background: #f5efe4;
      border-bottom: 2px solid var(--rtk-rule);
      padding: 7px 24px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .rtk-num-badge {
      font-family: 'Special Elite', monospace;
      font-size: 9px;
      letter-spacing: 2px;
      color: var(--rtk-sepia);
      text-transform: uppercase;
    }
    .rtk-num-line {
      flex: 1;
      height: 1px;
      background: var(--rtk-rule);
    }

    /* ── Card body ─────────────────────────────────────────────────────────── */
    #rtk-body {
      padding: 18px 24px 20px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0 20px;
    }

    /* ── Recipe title ──────────────────────────────────────────────────────── */
    #rtk-title-block {
      grid-column: 1 / -1;
      text-align: center;
      margin-bottom: 14px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--rtk-rule);
    }
    #rtk-emoji { font-size: 36px; display: block; margin-bottom: 4px; }
    #rtk-recipe-title {
      font-family: 'Playfair Display', serif;
      font-size: 26px;
      font-weight: 700;
      color: #2c1a08;
      line-height: 1.15;
      margin-bottom: 10px;
    }
    #rtk-meta {
      display: flex;
      justify-content: center;
      gap: 18px;
      font-family: 'Crimson Text', serif;
      font-size: 13px;
      color: var(--rtk-sepia);
    }
    #rtk-meta span::before { content: '◆ '; font-size: 8px; vertical-align: middle; }

    /* ── Sections ──────────────────────────────────────────────────────────── */
    .rtk-section-title {
      font-family: 'Special Elite', monospace;
      font-size: 9px;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      color: var(--rtk-red);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .rtk-section-title::after {
      content: '';
      flex: 1;
      height: 1px;
      background: var(--rtk-rule);
    }

    /* ── Ingredients ───────────────────────────────────────────────────────── */
    #rtk-ingredients { border-right: 1px solid var(--rtk-rule); padding-right: 16px; }
    .rtk-ingredient {
      font-family: 'Crimson Text', serif;
      font-size: 13.5px;
      color: #3a2510;
      margin-bottom: 5px;
      display: flex;
      gap: 7px;
      line-height: 1.35;
    }
    .rtk-ingredient .rtk-ing-emoji { flex-shrink: 0; font-size: 13px; }

    /* ── Method ────────────────────────────────────────────────────────────── */
    #rtk-method { padding-left: 4px; }
    .rtk-step {
      font-family: 'Crimson Text', serif;
      font-size: 13.5px;
      color: #3a2510;
      margin-bottom: 6px;
      display: flex;
      gap: 8px;
      line-height: 1.35;
    }
    .rtk-step-num {
      font-family: 'Special Elite', monospace;
      font-size: 11px;
      color: var(--rtk-red);
      flex-shrink: 0;
      width: 16px;
      margin-top: 1px;
    }

    /* ── Pairing + footer ──────────────────────────────────────────────────── */
    #rtk-pairing {
      grid-column: 1 / -1;
      background: #f5ece0;
      border-radius: 4px;
      padding: 9px 12px;
      margin: 10px 0 6px;
      font-family: 'Crimson Text', serif;
      font-style: italic;
      font-size: 13px;
      color: var(--rtk-sepia);
      border-left: 3px solid var(--rtk-amber);
    }

    #rtk-footer {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 4px;
    }
    #rtk-chef {
      font-family: 'Crimson Text', serif;
      font-size: 11px;
      color: var(--rtk-sepia);
      font-style: italic;
    }
    #rtk-debut {
      font-family: 'Special Elite', monospace;
      font-size: 9px;
      color: var(--rtk-sepia);
      letter-spacing: 1px;
      text-align: right;
    }

    /* ── CTA button ─────────────────────────────────────────────────────────── */
    #rtk-cta-row {
      grid-column: 1 / -1;
      text-align: center;
      margin-top: 14px;
      border-top: 2px solid var(--rtk-rule);
      padding-top: 14px;
    }
    #rtk-start-btn {
      font-family: 'Special Elite', monospace;
      font-size: 14px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #fff;
      background: var(--rtk-red);
      border: none;
      border-radius: 3px;
      padding: 12px 36px;
      cursor: pointer;
      box-shadow: 0 3px 0 #8a1525, 0 6px 20px rgba(196,30,58,0.35);
      transition: all 0.15s;
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    #rtk-start-btn:hover {
      background: #a81830;
      transform: translateY(-1px);
      box-shadow: 0 4px 0 #7a1120, 0 8px 28px rgba(196,30,58,0.45);
    }
    #rtk-start-btn:active {
      transform: translateY(1px);
      box-shadow: 0 2px 0 #8a1525, 0 4px 14px rgba(196,30,58,0.25);
    }
    #rtk-skip {
      display: block;
      margin-top: 8px;
      font-family: 'Crimson Text', serif;
      font-size: 11px;
      color: rgba(255,255,255,0.3);
      cursor: pointer;
      text-decoration: underline;
      border: none;
      background: none;
    }
    #rtk-skip:hover { color: rgba(255,255,255,0.55); }

    /* ── Decorative corner stamps ───────────────────────────────────────────── */
    .rtk-stamp {
      position: absolute;
      font-family: 'Special Elite', monospace;
      font-size: 8px;
      color: rgba(124,97,70,0.35);
      letter-spacing: 1px;
    }
    .rtk-stamp-tl { top: 10px; left: 14px; }
    .rtk-stamp-br { bottom: 10px; right: 14px; text-align: right; }
  `
  document.head.appendChild(style)
}

// ── Build the overlay DOM ─────────────────────────────────────────────────────

function buildOverlay() {
  const el = document.createElement('div')
  el.id = 'rtk-intro'

  // Steam particles (decorative)
  for (let i = 0; i < 12; i++) {
    const s = document.createElement('div')
    s.className = 'rtk-steam'
    const size = 2 + Math.random() * 8
    Object.assign(s.style, {
      left:             `${10 + Math.random() * 80}%`,
      bottom:           `${5 + Math.random() * 30}%`,
      height:           `${30 + Math.random() * 80}px`,
      width:            `${size}px`,
      animationDuration:`${4 + Math.random() * 6}s`,
      animationDelay:   `${Math.random() * 4}s`,
    })
    el.appendChild(s)
  }

  const d = RECIPE_DATA
  el.innerHTML += /* html */`
    <div id="rtk-card">

      <!-- Corner stamps -->
      <div class="rtk-stamp rtk-stamp-tl">RTK.001.A</div>
      <div class="rtk-stamp rtk-stamp-br">REDTEAMKITCHEN.COM</div>

      <!-- Red header -->
      <div id="rtk-header">
        <div id="rtk-episode">
          ${d.season}<br>${d.debut}
        </div>
        <div id="rtk-wordmark">
          🔴 Red Team Kitchen
          <span>A Cooking Lab for Big Ideas</span>
        </div>
        <div id="rtk-collab">in collaboration with Illinois State University</div>
      </div>

      <!-- Recipe number -->
      <div id="rtk-recipe-num">
        <div class="rtk-num-badge">Recipe No. ${d.number}</div>
        <div class="rtk-num-line"></div>
      </div>

      <!-- Body grid -->
      <div id="rtk-body">

        <!-- Title block -->
        <div id="rtk-title-block">
          <span id="rtk-emoji">${d.emoji}</span>
          <div id="rtk-recipe-title">${d.title}</div>
          <div id="rtk-meta">
            <span>Prep ${d.prepTime}</span>
            <span>Cook ${d.cookTime}</span>
            <span>Serves ${d.serves}</span>
          </div>
        </div>

        <!-- Ingredients -->
        <div id="rtk-ingredients">
          <div class="rtk-section-title">Ingredients</div>
          ${d.ingredients.map(i => `
            <div class="rtk-ingredient">
              <span class="rtk-ing-emoji">${i.emoji}</span>
              <span>${i.text}</span>
            </div>
          `).join('')}
        </div>

        <!-- Method -->
        <div id="rtk-method">
          <div class="rtk-section-title">Method</div>
          ${d.method.map((step, i) => `
            <div class="rtk-step">
              <span class="rtk-step-num">${i + 1}.</span>
              <span>${step}</span>
            </div>
          `).join('')}
        </div>

        <!-- Pairing note -->
        <div id="rtk-pairing">
          <strong>Chef's pairing:</strong> ${d.pairing}
        </div>

        <!-- Footer -->
        <div id="rtk-footer">
          <div id="rtk-chef">Chef: ${d.chef}</div>
          <div id="rtk-debut">${d.debut}</div>
        </div>

        <!-- CTA -->
        <div id="rtk-cta-row">
          <button id="rtk-start-btn">🍳 &nbsp;Start Cooking</button>
          <button id="rtk-skip">Skip intro</button>
        </div>

      </div><!-- /body -->
    </div><!-- /card -->
  `
  return el
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Show the Red Team Kitchen recipe intro overlay.
 * Resolves immediately if already seen this session, otherwise
 * waits for the user to click "Start Cooking" or "Skip intro".
 */
export function showRecipeIntro() {
  // Only show once per session
  if (sessionStorage.getItem('rtk_intro_seen')) return Promise.resolve()

  return new Promise(resolve => {
    injectStyles()
    const overlay = buildOverlay()
    document.body.appendChild(overlay)

    function dismiss() {
      sessionStorage.setItem('rtk_intro_seen', '1')
      overlay.classList.add('hiding')
      setTimeout(() => {
        overlay.remove()
        resolve()
      }, 580)
    }

    overlay.querySelector('#rtk-start-btn').addEventListener('click', dismiss)
    overlay.querySelector('#rtk-skip').addEventListener('click', dismiss)

    // Keyboard: Enter or Space dismisses
    function onKey(e) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') {
        document.removeEventListener('keydown', onKey)
        dismiss()
      }
    }
    document.addEventListener('keydown', onKey)
  })
}

/**
 * Reset intro (for testing — call from browser console: RTK.resetIntro())
 */
export function resetIntro() {
  sessionStorage.removeItem('rtk_intro_seen')
}

// Expose for console debugging
if (typeof window !== 'undefined') {
  window.RTK = { resetIntro }
}
