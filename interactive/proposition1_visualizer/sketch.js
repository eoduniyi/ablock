/**
 * PROPOSITION 1 — Progressive Disclosure, Typography & Motion
 *
 * Fixed typewriter: parent .proof-line never uses opacity to hide content.
 * Instead, individual .type-char spans start at opacity:0 and illuminate
 * to opacity:1 as they're typed. Past lines get class .past (all chars
 * forced visible). Future lines get class .latent (faint watermark).
 *
 * 5 distinct palettes: Vellum, Darkroom, Moss, Clay, Indigo.
 */

// ============================================================================
// STATE
// ============================================================================
let currentStep = 0;
const TOTAL_STEPS = 7;
let autoAdvance = true;
let autoTimer = null;
const AUTO_INTERVAL_MS = 6500;

let particles = [];
const NUM_PARTICLES = 130;
let qedActive = false;
let qedRings = [];
let regimeBlend = 0.0;

let tombstonePos = { x: 0, y: 0 };
let plateBounds = { left: 0, top: 0, right: 0, bottom: 0, cx: 0, cy: 0, halfW: 0, halfH: 0 };

// ============================================================================
// PALETTES
// ============================================================================
const PALETTES = {
  vellum: {
    name: 'VELLUM', isDark: false, swatch: '#92400e',
    bg: [247, 245, 240, 185], grid: [0, 0, 0, 6],
    chords: [146, 64, 14], streamlines: [2, 132, 199],
    pMeasure: [2, 132, 199], pGraph: [146, 64, 14], locked: [146, 64, 14]
  },
  darkroom: {
    name: 'DARKROOM', isDark: true, swatch: '#dc2626',
    bg: [10, 10, 12, 185], grid: [255, 255, 255, 5],
    chords: [220, 38, 38], streamlines: [180, 180, 180],
    pMeasure: [180, 180, 180], pGraph: [220, 38, 38], locked: [220, 38, 38]
  },
  moss: {
    name: 'MOSS', isDark: true, swatch: '#86efac',
    bg: [26, 31, 26, 185], grid: [134, 239, 172, 8],
    chords: [134, 239, 172], streamlines: [52, 211, 153],
    pMeasure: [52, 211, 153], pGraph: [134, 239, 172], locked: [134, 239, 172]
  },
  clay: {
    name: 'CLAY', isDark: false, swatch: '#c2410c',
    bg: [240, 230, 219, 185], grid: [0, 0, 0, 7],
    chords: [194, 65, 12], streamlines: [120, 53, 15],
    pMeasure: [120, 53, 15], pGraph: [194, 65, 12], locked: [194, 65, 12]
  },
  indigo: {
    name: 'INDIGO', isDark: true, swatch: '#a78bfa',
    bg: [12, 10, 26, 185], grid: [129, 140, 248, 8],
    chords: [167, 139, 250], streamlines: [99, 102, 241],
    pMeasure: [99, 102, 241], pGraph: [167, 139, 250], locked: [167, 139, 250]
  }
};

const THEME_KEYS = ['vellum', 'darkroom', 'moss', 'clay', 'indigo'];
let currentTheme = 'vellum';

function pal() { return PALETTES[currentTheme]; }

function setTheme(key) {
  if (!PALETTES[key]) key = 'vellum';
  currentTheme = key;
  document.documentElement.setAttribute('data-theme', key);
  const sw = document.getElementById('theme-swatch');
  const tx = document.getElementById('theme-text');
  if (tx) tx.innerText = PALETTES[key].name;
  if (sw) sw.style.color = PALETTES[key].swatch;
}

function cycleTheme() {
  const i = (THEME_KEYS.indexOf(currentTheme) + 1) % THEME_KEYS.length;
  setTheme(THEME_KEYS[i]);
}

// ============================================================================
// PERSPECTIVE LENSES (SYSTEMS vs MECHANISTIC)
// ============================================================================
const LENSES = {
  systems: {
    key: 'systems',
    label: 'SYSTEMS',
    subtitle: '(Linear-Time Particle Updates)',
    tooltip: 'Systems lens: Bypasses the O(N²) all-pairs bottleneck for real-time parallel scaling.'
  },
  mechanistic: {
    key: 'mechanistic',
    label: 'MECHANISTIC',
    subtitle: '(Independent Particle Dynamics)',
    tooltip: 'Mechanistic lens: Local Clifford score rotors and neural gradients decouple pairwise graph edges.'
  }
};

let currentLens = 'systems';

function toggleLens() {
  currentLens = currentLens === 'systems' ? 'mechanistic' : 'systems';
  const cfg = LENSES[currentLens];
  const navTxt = document.getElementById('lens-text');
  const subTxt = document.getElementById('prop-subtitle-text');
  const badge = document.getElementById('prop-lens-badge');
  const btn = document.getElementById('btn-prop-subtitle');

  if (navTxt) navTxt.textContent = cfg.label;
  if (badge) badge.textContent = cfg.label;
  if (btn) btn.title = `Click to toggle perspective (Hotkey: M. Currently: ${cfg.tooltip})`;

  if (subTxt) {
    subTxt.style.opacity = '0';
    setTimeout(() => {
      subTxt.textContent = cfg.subtitle;
      subTxt.style.opacity = '1';
    }, 150);
  }

  if (currentStep === 0) updateTelemetry(false);
}

function setLens(lensKey) {
  if (LENSES[lensKey] && currentLens !== lensKey) toggleLens();
}
// AUDIO
// ============================================================================
let audioCtx = null;

function initAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

function playTick() {
  try {
    initAudio();
    const now = audioCtx.currentTime;
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = 'triangle';
    o.frequency.setValueAtTime(1200 + Math.random() * 400, now);
    o.frequency.exponentialRampToValueAtTime(400, now + 0.015);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(0.012, now + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, now + 0.018);
    o.connect(g); g.connect(audioCtx.destination);
    o.start(now); o.stop(now + 0.02);
  } catch(e){}
}

function playChime(step) {
  try {
    initAudio();
    const now = audioCtx.currentTime;
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    const freqs = [293.66, 329.63, 369.99, 440, 493.88, 587.33, 659.25];
    const f = freqs[step % freqs.length];
    o.type = 'sine';
    o.frequency.setValueAtTime(f, now);
    o.frequency.exponentialRampToValueAtTime(f * 0.99, now + 0.45);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(0.05, now + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, now + 0.44);
    o.connect(g); g.connect(audioCtx.destination);
    o.start(now); o.stop(now + 0.46);
  } catch(e){}
}

function playQED() {
  try {
    initAudio();
    const now = audioCtx.currentTime;
    const o1 = audioCtx.createOscillator();
    const o2 = audioCtx.createOscillator();
    const flt = audioCtx.createBiquadFilter();
    const g = audioCtx.createGain();
    o1.type = 'sine'; o1.frequency.setValueAtTime(220, now);
    o1.frequency.exponentialRampToValueAtTime(110, now + 2.2);
    o2.type = 'triangle'; o2.frequency.setValueAtTime(329.63, now);
    o2.frequency.exponentialRampToValueAtTime(164.81, now + 2.4);
    flt.type = 'lowpass'; flt.frequency.setValueAtTime(1200, now);
    flt.frequency.exponentialRampToValueAtTime(200, now + 2.4);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.linearRampToValueAtTime(0.16, now + 0.04);
    g.gain.exponentialRampToValueAtTime(0.0001, now + 2.8);
    o1.connect(flt); o2.connect(flt); flt.connect(g); g.connect(audioCtx.destination);
    o1.start(now); o2.start(now); o1.stop(now + 2.85); o2.stop(now + 2.85);
  } catch(e){}
}

// ============================================================================
// KATEX
// ============================================================================
function renderMath(s) {
  if (typeof katex !== 'undefined') {
    try { return katex.renderToString(s, { throwOnError: false, displayMode: false }); }
    catch(e) { return s; }
  }
  return s;
}

function renderStaticMath() {
  if (typeof renderMathInElement !== 'undefined') {
    try {
      renderMathInElement(document.getElementById('proof-plate'), {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    } catch(e) {}
  }
}

// ============================================================================
// TYPEWRITER ENGINE
// ============================================================================
let typingTimer = null;
let isTyping = false;
let typingStep = -1;

/**
 * Walk the DOM tree of each proof-line and wrap every text character
 * in a <span class="type-char">. KaTeX .katex spans get tagged as
 * type-char type-math (treated as single atomic units).
 */
function prepareChars() {
  function walk(node, isBlockStart = true) {
    // If this is a proof-line container, only walk its actual content elements (p, h1), completely stripping whitespace text nodes
    if (node.classList && node.classList.contains('proof-line')) {
      const kids = Array.from(node.childNodes);
      for (const kid of kids) {
        if (kid.nodeType === Node.TEXT_NODE) {
          kid.remove();
        } else if (kid.nodeType === Node.ELEMENT_NODE) {
          walk(kid, true);
        }
      }
      return;
    }

    const kids = Array.from(node.childNodes);
    let atStart = isBlockStart;

    for (let k = 0; k < kids.length; k++) {
      const kid = kids[k];
      if (kid.nodeType === Node.TEXT_NODE) {
        let txt = kid.textContent.replace(/\s+/g, ' ');
        if (atStart) {
          txt = txt.trimStart();
        }
        // Always trim end on the final child of any block element
        if (k === kids.length - 1) {
          txt = txt.trimEnd();
        }
        if (!txt) {
          kid.remove();
          continue;
        }

        atStart = false;

        const frag = document.createDocumentFragment();
        for (let i = 0; i < txt.length; i++) {
          const ch = txt[i];
          const sp = document.createElement('span');
          if (ch === ' ') {
            sp.className = 'type-char type-space';
            sp.textContent = ch;
          } else if (',.;:!?)]}'.includes(ch)) {
            sp.className = 'type-char type-punct';
            // Prepend Unicode Word Joiner (\u2060) to prevent dangling punctuation at line start
            sp.textContent = '\u2060' + ch;
          } else {
            sp.className = 'type-char';
            sp.textContent = ch;
          }
          frag.appendChild(sp);
        }
        kid.parentNode.replaceChild(frag, kid);
      } else if (kid.nodeType === Node.ELEMENT_NODE) {
        atStart = false;
        if (kid.classList.contains('katex')) {
          kid.classList.add('type-char', 'type-math');
        } else if (kid.classList.contains('tombstone-wrap')) {
          kid.classList.add('type-char');
        } else if (kid.classList.contains('prop-subtitle-btn') || kid.id === 'btn-prop-subtitle') {
          continue;
        } else {
          walk(kid, false);
        }
      }
    }
  }

  for (let i = 0; i <= 5; i++) {
    const el = document.querySelector(`.proof-line[data-step="${i}"]`);
    if (el) walk(el, true);
  }
}

function revealAllChars(lineEl) {
  lineEl.querySelectorAll('.type-char').forEach(c => c.classList.add('typed'));
}

function hideAllChars(lineEl) {
  lineEl.querySelectorAll('.type-char').forEach(c => c.classList.remove('typed'));
}

function clearActiveCursor() {
  document.querySelectorAll('.has-cursor').forEach(el => el.classList.remove('has-cursor'));
  document.querySelectorAll('.has-cursor-before').forEach(el => el.classList.remove('has-cursor-before'));
  document.querySelectorAll('.type-cursor').forEach(el => el.remove());
}

function typeLine(step) {
  stopTyping();
  clearActiveCursor();
  if (step === 6) return;

  const lineEl = document.querySelector(`.proof-line[data-step="${step}"]`);
  if (!lineEl) return;

  const chars = lineEl.querySelectorAll('.type-char');
  if (chars.length === 0) return;

  // Reset
  hideAllChars(lineEl);

  isTyping = true;
  typingStep = step;
  let idx = 0;

  // Position cursor indicator before first character begins
  if (chars[0]) chars[0].classList.add('has-cursor-before');

  function tick() {
    if (!isTyping || typingStep !== step) {
      clearActiveCursor();
      return;
    }

    if (idx < chars.length) {
      if (idx === 0 && chars[0]) {
        chars[0].classList.remove('has-cursor-before');
      }
      const prev = chars[idx - 1];
      if (prev) prev.classList.remove('has-cursor');

      const c = chars[idx];
      c.classList.add('typed');
      c.classList.add('has-cursor');

      // Subtle pen contact sound every few characters
      if (idx % 4 === 0) playTick();

      idx++;

      // Cadence
      let delay = 18;
      if (c.classList.contains('type-math')) delay = 50;
      else if (c.classList.contains('type-space')) delay = 6;
      else if (c.textContent === '.' || c.textContent === ',') delay = 55;
      else if (c.textContent === '(' || c.textContent === ')') delay = 30;

      typingTimer = setTimeout(tick, delay);
    } else {
      // Done typing: immediately clear cursor so it never hangs or expands vertical space
      isTyping = false;
      clearActiveCursor();
    }
  }

  tick();
}

function finishTyping() {
  if (!isTyping || typingStep < 0) return;
  const lineEl = document.querySelector(`.proof-line[data-step="${typingStep}"]`);
  if (lineEl) revealAllChars(lineEl);
  clearActiveCursor();
  isTyping = false;
  clearTimeout(typingTimer);
}

function stopTyping() {
  isTyping = false;
  clearTimeout(typingTimer);
  clearActiveCursor();
}

// ============================================================================
// PARTICLES
// ============================================================================
class Particle {
  constructor() {
    this.x = random(width);
    this.y = random(height);
    this.vx = random(-0.5, 0.5);
    this.vy = random(-0.5, 0.5);
    this.mass = random(0.9, 1.2);
    this.maxSpd = random(1.5, 2.3);
    this.r = random(1.7, 2.5);
    this.chi = random() > 0.5 ? 1 : -1;
    this.locked = false;
    this.oAngle = random(TWO_PI);
    this.oRad = random(20, 55);
  }

  update() {
    if (this.locked) {
      this.oAngle += 0.016 * this.chi;
      this.x = lerp(this.x, tombstonePos.x + Math.cos(this.oAngle) * this.oRad, 0.08);
      this.y = lerp(this.y, tombstonePos.y + Math.sin(this.oAngle) * this.oRad, 0.08);
      this.vx = this.vy = 0;
      return;
    }

    let fx = 0, fy = 0;
    const tdx = tombstonePos.x - this.x;
    const tdy = tombstonePos.y - this.y;
    const td = Math.hypot(tdx, tdy) + 0.1;
    const gx = tdx / td, gy = tdy / td;
    const rx = -gy, ry = gx;

    // Clifford flow
    const sig = 0.85 * this.chi, kap = 0.20;
    const fs = 85 / (td + 55);
    const cfx = (sig * rx + kap * gx) * fs;
    const cfy = (sig * ry + kap * gy) * fs;

    // Graph flow
    const na = noise(this.x * 0.004, this.y * 0.004, frameCount * 0.008) * TWO_PI * 2;
    const gfx = gx * 0.12 + Math.cos(na) * 0.35;
    const gfy = gy * 0.12 + Math.sin(na) * 0.35;

    fx = lerp(cfx, gfx, regimeBlend);
    fy = lerp(cfy, gfy, regimeBlend);

    // QED pull
    if (qedActive || currentStep === 6) {
      fx = gx * 1.6 + rx * 0.4 * this.chi;
      fy = gy * 1.6 + ry * 0.4 * this.chi;
      if (td < 52) {
        this.locked = true;
        this.oRad = Math.max(18, td);
        this.oAngle = Math.atan2(this.y - tombstonePos.y, this.x - tombstonePos.x);
      }
    }

    // Plate deflection
    const d = deflect(this.x, this.y, this.vx, this.vy);
    fx += d.fx; fy += d.fy;

    // Mouse swirl
    const mdx = mouseX - this.x, mdy = mouseY - this.y;
    const md = Math.hypot(mdx, mdy) + 0.1;
    if (md < 110) {
      const mf = (1 - md / 110) * 0.8;
      fx += (-mdy / md) * mf;
      fy += (mdx / md) * mf;
    }

    // Newtonian integration
    let ax = fx / this.mass, ay = fy / this.mass;
    const am = Math.hypot(ax, ay);
    if (am > 1.8) { ax = ax / am * 1.8; ay = ay / am * 1.8; }

    const damp = lerp(0.965, 0.94, regimeBlend);
    this.vx = (this.vx + ax * 0.3) * damp;
    this.vy = (this.vy + ay * 0.3) * damp;

    const spd = Math.hypot(this.vx, this.vy);
    if (spd > this.maxSpd) {
      this.vx = this.vx / spd * this.maxSpd;
      this.vy = this.vy / spd * this.maxSpd;
    }

    this.x += this.vx;
    this.y += this.vy;

    if (this.x < -16) this.x = width + 16;
    if (this.x > width + 16) this.x = -16;
    if (this.y < -16) this.y = height + 16;
    if (this.y > height + 16) this.y = -16;
  }

  draw() {
    noStroke();
    const p = pal();
    if (this.locked) {
      fill(p.locked[0], p.locked[1], p.locked[2], 230);
    } else {
      fill(
        lerp(p.pMeasure[0], p.pGraph[0], regimeBlend),
        lerp(p.pMeasure[1], p.pGraph[1], regimeBlend),
        lerp(p.pMeasure[2], p.pGraph[2], regimeBlend),
        160
      );
    }
    circle(this.x, this.y, this.r * 2);
  }
}

function deflect(px, py, vx, vy) {
  const pad = 34;
  const hw = plateBounds.halfW + pad, hh = plateBounds.halfH + pad;
  const drx = px - plateBounds.cx, dry = py - plateBounds.cy;
  const dx = Math.abs(drx) - hw, dy = Math.abs(dry) - hh;

  if (dx < 0 && dy < 0) {
    return { fx: dx > dy ? Math.sign(drx) * 1.4 : 0, fy: dy >= dx ? Math.sign(dry) * 1.4 : 0 };
  }

  const ox = Math.max(dx, 0), oy = Math.max(dy, 0);
  const dist = Math.hypot(ox, oy);
  if (dist < 48) {
    const t = 1 - dist / 48;
    const s = t * t * (3 - 2 * t) * 0.9;
    let nx = ox > 0 ? Math.sign(drx) * ox / (dist + 1e-4) : 0;
    let ny = oy > 0 ? Math.sign(dry) * oy / (dist + 1e-4) : 0;
    let tx = -ny, ty = nx;
    if (vx * tx + vy * ty < 0) { tx = -tx; ty = -ty; }
    return { fx: nx * s * 1.25 + tx * s * 0.75, fy: ny * s * 1.25 + ty * s * 0.75 };
  }
  return { fx: 0, fy: 0 };
}

// ============================================================================
// P5 CANVAS
// ============================================================================
function setup() {
  const c = createCanvas(windowWidth, windowHeight);
  c.parent('canvas-container');
  pixelDensity(window.devicePixelRatio || 1);

  for (let i = 0; i < NUM_PARTICLES; i++) particles.push(new Particle());

  updateCoords();
  setupInteractions();

  renderStaticMath();
  prepareChars();

  setTheme('vellum');
  setStep(0, false);
  startAutoAdvance();
}

function windowResized() { resizeCanvas(windowWidth, windowHeight); updateCoords(); }

function updateCoords() {
  const te = document.getElementById('tombstone');
  const pe = document.getElementById('proof-plate');
  if (te) { const r = te.getBoundingClientRect(); tombstonePos.x = r.left + r.width / 2; tombstonePos.y = r.top + r.height / 2; }
  if (pe) { const r = pe.getBoundingClientRect(); plateBounds = { left: r.left, top: r.top, right: r.right, bottom: r.bottom, cx: r.left + r.width / 2, cy: r.top + r.height / 2, halfW: r.width / 2, halfH: r.height / 2 }; }
}

function draw() {
  const p = pal();
  background(p.bg[0], p.bg[1], p.bg[2], p.bg[3]);

  const tgt = (currentStep === 1 || currentStep === 3) ? 1.0 : 0.0;
  regimeBlend = lerp(regimeBlend, tgt, 0.035);

  // Grid
  strokeWeight(1);
  stroke(p.grid[0], p.grid[1], p.grid[2], p.grid[3]);
  for (let x = 0; x < width; x += 64) line(x, 0, x, height);
  for (let y = 0; y < height; y += 64) line(0, y, width, y);

  // Shockwaves
  for (let i = qedRings.length - 1; i >= 0; i--) {
    const r = qedRings[i];
    r.radius += 7; r.alpha -= 2.8;
    noFill(); stroke(p.locked[0], p.locked[1], p.locked[2], r.alpha);
    strokeWeight(r.weight); circle(r.x, r.y, r.radius * 2);
    if (r.alpha <= 0) qedRings.splice(i, 1);
  }

  // Graph chords
  if (regimeBlend > 0.02) {
    strokeWeight(0.85);
    const th = 50, t2 = th * th;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d2 = dx * dx + dy * dy;
        if (d2 < t2) {
          const d = Math.sqrt(d2);
          const a = map(d, 0, th, p.isDark ? 115 : 75, 0) * regimeBlend;
          stroke(p.chords[0], p.chords[1], p.chords[2], a);
          line(particles[i].x, particles[i].y, particles[j].x, particles[j].y);
        }
      }
    }
  }

  // Streamlines
  if (regimeBlend < 0.98 && currentStep >= 2 && currentStep < 6) {
    strokeWeight(0.95);
    const w = 1 - regimeBlend;
    for (let i = 0; i < particles.length; i += 4) {
      const pt = particles[i];
      const dx = tombstonePos.x - pt.x, dy = tombstonePos.y - pt.y;
      const d = Math.hypot(dx, dy) + 0.1;
      const a = (p.isDark ? 50 : 42) * w;
      stroke(p.streamlines[0], p.streamlines[1], p.streamlines[2], a);
      line(pt.x, pt.y, pt.x + (-dy / d) * 12, pt.y + (dx / d) * 12);
    }
  }

  for (const pt of particles) { pt.update(); pt.draw(); }
}

// ============================================================================
// STEP PROGRESSION
// ============================================================================
function setStep(newStep, playAudio = true) {
  finishTyping();
  currentStep = constrain(newStep, 0, TOTAL_STEPS - 1);

  if (autoAdvance) {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = setInterval(nextStep, AUTO_INTERVAL_MS);
  } else {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = null;
  }

  const lines = document.querySelectorAll('.proof-line');
  lines.forEach((el, i) => {
    el.classList.remove('visible', 'current', 'past', 'latent');
    if (i < currentStep) {
      el.classList.add('visible', 'past');
      revealAllChars(el);
    } else if (i === currentStep) {
      el.classList.add('visible', 'current');
      // Don't reveal chars yet — typeLine will animate them
    } else {
      el.classList.add('latent');
      hideAllChars(el);
    }
  });

  // Progress pips
  document.querySelectorAll('.progress-pip').forEach((pip, i) => {
    pip.classList.toggle('active', i <= currentStep);
  });

  // Step counter
  const romans = ['I','II','III','IV','V','VI','VII'];
  const ctr = document.getElementById('step-counter');
  if (ctr) ctr.innerText = `STEP ${romans[currentStep]} / VII`;

  updateTelemetry(true);
  updateCoords();

  if (currentStep === 6) {
    // Tombstone: just reveal the tombstone line instantly
    const tombLine = document.querySelector('.proof-line[data-step="6"]');
    if (tombLine) revealAllChars(tombLine);
    triggerQED();
  } else {
    typeLine(currentStep);
    if (playAudio) {
      playChime(currentStep);
      for (const pt of particles) pt.locked = false;
    }
  }
}

function nextStep() {
  if (isTyping) { finishTyping(); return; }
  setStep(currentStep < TOTAL_STEPS - 1 ? currentStep + 1 : 0);
}

function prevStep() {
  finishTyping();
  if (currentStep > 0) setStep(currentStep - 1);
}

function triggerQED() {
  playQED();
  updateCoords();
  qedRings.push({ x: tombstonePos.x, y: tombstonePos.y, radius: 10, alpha: 240, weight: 2.5 });
  qedRings.push({ x: tombstonePos.x, y: tombstonePos.y, radius: 2, alpha: 180, weight: 1.5 });
  const te = document.getElementById('tombstone');
  if (te) te.classList.add('qed-active');
  setTimeout(() => { if (te) te.classList.remove('qed-active'); }, 4500);
}

// ============================================================================
// TELEMETRY
// ============================================================================
function updateTelemetry(animate) {
  const box = document.getElementById('telemetry-info');
  const reg = document.getElementById('telemetry-regime');
  const cost = document.getElementById('telemetry-cost');
  if (!reg || !cost) return;

  const n = particles.length;
  const ev = ((n * (n - 1)) / 2).toLocaleString();

  const apply = () => {
    switch (currentStep) {
      case 0:
        reg.innerHTML = currentLens === 'systems' ? "Systems Lens" : "Mechanistic Lens";
        cost.innerHTML = currentLens === 'systems'
          ? "Linear-time updates (Hotkey: <kbd>M</kbd> to toggle)"
          : "Independent score rotors (Hotkey: <kbd>M</kbd> to toggle)";
        break;
      case 1: reg.innerHTML = renderMath("G_\\varepsilon \\;\\text{[Metric Graph]}"); cost.innerHTML = `${renderMath("\\binom{N}{2}=" + ev)} all-pairs`; break;
      case 2: reg.innerHTML = renderMath("\\text{A-Block } (\\mathcal{T})"); cost.innerHTML = `Decoupled measure: strictly ${renderMath("\\mathcal{O}(N)=" + n)}`; break;
      case 3: reg.innerHTML = renderMath("\\mathbf{A}_{ij}\\;\\text{[Adjacency]}"); cost.innerHTML = `Quadratic ${renderMath("\\mathcal{O}(N^2)")}`; break;
      case 4: reg.innerHTML = renderMath("\\mathbf{f}_i\\;\\text{[Clifford Rotor]}"); cost.innerHTML = `Linear ${renderMath("\\mathcal{O}(N)")}`; break;
      case 5: reg.innerHTML = renderMath("\\textstyle\\sum_k\\;\\text{[Ensemble]}"); cost.innerHTML = renderMath("\\mathcal{O}(N_{\\text{total}})"); break;
      case 6: reg.innerHTML = renderMath("\\mu_\\infty\\;\\text{[Equilibrium]}"); cost.innerHTML = "Halmos stationarity"; break;
    }
  };

  if (animate && box) {
    box.classList.add('fading');
    setTimeout(() => { apply(); box.classList.remove('fading'); }, 120);
  } else {
    apply();
  }
}

// ============================================================================
// AUTO-ADVANCE & THEME
// ============================================================================
function updatePlayButtonUI() {
  const icon = document.getElementById('play-icon');
  const txt = document.getElementById('play-text') || document.getElementById('auto-text');
  const btn = document.getElementById('btn-play') || document.getElementById('btn-auto');
  if (icon) icon.innerText = autoAdvance ? '⏸' : '▶';
  if (txt) txt.innerText = autoAdvance ? 'PAUSE' : 'PLAY';
  if (btn) {
    btn.title = autoAdvance ? 'Pause Auto-Advance (Hotkey: Space or P)' : 'Play Auto-Advance (Hotkey: Space or P)';
    btn.classList.toggle('is-paused', !autoAdvance);
  }
}

function startAutoAdvance() {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
  if (autoAdvance) {
    autoTimer = setInterval(nextStep, AUTO_INTERVAL_MS);
  }
  updatePlayButtonUI();
}

function toggleAuto(forceState) {
  if (typeof forceState === 'boolean') {
    autoAdvance = forceState;
  } else {
    autoAdvance = !autoAdvance;
  }
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
  if (autoAdvance) {
    startAutoAdvance();
  }
  updatePlayButtonUI();
}
window.toggleAuto = toggleAuto;

// ============================================================================
// INTERACTIONS
// ============================================================================
function setupInteractions() {
  const plate = document.getElementById('proof-plate');
  const btnT = document.getElementById('btn-theme');
  const btnL = document.getElementById('btn-lens');
  const btnSub = document.getElementById('btn-prop-subtitle');
  const btnPlay = document.getElementById('btn-play') || document.getElementById('btn-auto');
  const btnR = document.getElementById('btn-replay');
  const tomb = document.getElementById('tombstone');

  if (plate) plate.addEventListener('click', e => {
    if (e.target.closest('#tombstone')) { e.stopPropagation(); triggerQED(); return; }
    if (e.target.closest('#btn-prop-subtitle')) return; // handled separately
    nextStep();
  });
  if (tomb) tomb.addEventListener('click', e => { e.stopPropagation(); triggerQED(); });
  if (btnT) btnT.addEventListener('click', e => { e.stopPropagation(); cycleTheme(); });
  if (btnL) btnL.addEventListener('click', e => { e.stopPropagation(); toggleLens(); });
  if (btnSub) btnSub.addEventListener('click', e => { e.stopPropagation(); toggleLens(); });
  if (btnPlay) btnPlay.addEventListener('click', e => { e.stopPropagation(); toggleAuto(); });
  if (btnR) btnR.addEventListener('click', e => { e.stopPropagation(); setStep(0); toggleAuto(true); });

  document.querySelectorAll('.progress-pip').forEach(pip => {
    pip.addEventListener('click', e => { e.stopPropagation(); setStep(+pip.dataset.step); });
  });

  window.addEventListener('keydown', e => {
    if (e.code === 'Space' || e.key === 'p' || e.key === 'P') {
      e.preventDefault();
      toggleAuto();
    }
    else if (e.key === 'ArrowRight' || e.key === 'Enter') { e.preventDefault(); nextStep(); }
    else if (e.key === 'ArrowLeft' || e.key === 'Backspace') { e.preventDefault(); prevStep(); }
    else if (e.key === 'c' || e.key === 'C') cycleTheme();
    else if (e.key === 'l' || e.key === 'L') toggleLightDark();
    else if (e.key === 'm' || e.key === 'M') toggleLens();
    else if (e.key === 'r' || e.key === 'R') { setStep(0); toggleAuto(true); }
    else if (e.key === 'a' || e.key === 'A') toggleAuto();
  });
}
