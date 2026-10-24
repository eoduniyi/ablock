let gridCols = 6;
let gridRows = 4;
let numPerCell = 100;
let num = gridCols * gridRows * numPerCell;
let ballCollection = []; 
let theta = 10;
let letterGraphics;
let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
let gridLetters = [];

// Modes: 'breathing', 'stable', 'dissolve'
let currentMode = 'breathing';
let pulseCycle = 0;
let isContained = true; // Default: contained inside respective dishes

let gridTop = 45;

let wordBank = ["MAGIC", "LOVE", "TETA", "THANKS", "LIGHT", "DARK", "FULL_SEQ", "FULL_SHUF"];
let currentWordIdx = 0;
let activeWord = "MAGIC";
let isCustomPaletteMode = true; // Default: unique color theme per word! Toggle with M key.
let textOrientation = 'LR'; // 'LR' (Left-to-Right Horizontal) or 'TD' (Top-to-Down Vertical). Toggle with O key.

const specimenPalettes = {
  "MAGIC": {
    name: "VIOLET SPECTRUM",
    bg: [15, 10, 24],
    agar: [24, 15, 36],
    border: [110, 50, 160],
    nodeCore: [255, 220, 255],
    nodeOuter: [195, 100, 255],
    glassRim: [210, 150, 255],
    accent: [220, 130, 255],
    hud: [220, 140, 255]
  },
  "LOVE": {
    name: "ROSE CORAL",
    bg: [22, 9, 14],
    agar: [34, 15, 22],
    border: [160, 50, 85],
    nodeCore: [255, 220, 230],
    nodeOuter: [245, 80, 120],
    glassRim: [255, 140, 170],
    accent: [255, 110, 150],
    hud: [255, 130, 160]
  },
  "TETA": {
    name: "GOLDEN AMBER",
    bg: [20, 15, 9],
    agar: [32, 24, 14],
    border: [150, 110, 45],
    nodeCore: [255, 240, 180],
    nodeOuter: [235, 175, 55],
    glassRim: [255, 215, 130],
    accent: [245, 185, 65],
    hud: [255, 195, 75]
  },
  "THANKS": {
    name: "EMERALD BIO",
    bg: [9, 20, 15],
    agar: [14, 32, 24],
    border: [45, 145, 95],
    nodeCore: [210, 255, 230],
    nodeOuter: [45, 225, 145],
    glassRim: [130, 255, 195],
    accent: [65, 235, 155],
    hud: [85, 245, 165]
  },
  "LIGHT": {
    name: "SOLAR LUMEN",
    bg: [235, 240, 245],
    agar: [215, 225, 235],
    border: [80, 130, 190],
    nodeCore: [255, 255, 255],
    nodeOuter: [20, 120, 220],
    glassRim: [40, 100, 180],
    accent: [30, 110, 200],
    hud: [20, 40, 70]
  },
  "DARK": {
    name: "OBSIDIAN INDIGO",
    bg: [5, 6, 12],
    agar: [10, 12, 22],
    border: [55, 65, 120],
    nodeCore: [220, 230, 255],
    nodeOuter: [90, 110, 240],
    glassRim: [130, 150, 250],
    accent: [110, 130, 255],
    hud: [140, 160, 255]
  },
  "DEFAULT": {
    name: "CYBER MONO",
    bg: [12, 14, 19],
    agar: [15, 20, 28],
    border: [35, 65, 95],
    nodeCore: [240, 250, 255],
    nodeOuter: [120, 190, 255],
    glassRim: [180, 230, 255],
    accent: [100, 180, 255],
    hud: [140, 200, 255]
  }
};

function getActivePalette() {
  if (!isCustomPaletteMode) return specimenPalettes["DEFAULT"];
  return specimenPalettes[activeWord] || specimenPalettes["DEFAULT"];
}

function setup() {
  createCanvas(600, 600 + gridTop);
  pixelDensity(1); // Ensure 1:1 pixel mapping for get() and pixels[]
  
  letterGraphics = createGraphics(width, height);
  letterGraphics.pixelDensity(1);
  
  setWord(activeWord);
}

function setWord(word) {
  activeWord = word.toUpperCase().replace(/[^A-Z]/g, '');
  if (activeWord.length === 0) {
    gridCols = (textOrientation === 'LR') ? 4 : 1;
    gridRows = (textOrientation === 'LR') ? 1 : 4;
    gridLetters = [""];
    createStuff();
    return;
  }

  // Set number of petri dishes equal to the length of the word in LR (row) or TD (column)
  if (textOrientation === 'LR') {
    gridCols = activeWord.length;
    gridRows = 1;
  } else {
    gridCols = 1;
    gridRows = activeWord.length;
  }
  gridLetters = activeWord.split('');
  createStuff();
}

function toggleOrientation() {
  textOrientation = (textOrientation === 'LR') ? 'TD' : 'LR';
  let target = wordBank[currentWordIdx];
  if (target === "FULL_SEQ") {
    setFullAlphabetSequential();
  } else if (target === "FULL_SHUF") {
    setFullAlphabetShuffled();
  } else {
    setWord(activeWord);
  }
}

function setFullAlphabetSequential() {
  activeWord = "FULL ALPHABET (A-Z)";
  gridCols = 6;
  gridRows = 5;
  gridLetters = new Array(gridCols * gridRows).fill("");
  if (textOrientation === 'TD') {
    // Column-first order for TD mode
    let idx = 0;
    for (let c = 0; c < gridCols; c++) {
      for (let r = 0; r < gridRows; r++) {
        if (idx < alphabet.length) {
          gridLetters[r * gridCols + c] = alphabet[idx];
          idx++;
        }
      }
    }
  } else {
    for (let i = 0; i < alphabet.length; i++) {
      gridLetters[i] = alphabet[i];
    }
  }
  createStuff();
}

function setFullAlphabetShuffled() {
  activeWord = "RANDOM ALPHABET (A-Z)";
  gridCols = 6;
  gridRows = 5;
  let pool = alphabet.split('');
  // Fisher-Yates shuffle for all 26 letters A-Z
  for (let i = pool.length - 1; i > 0; i--) {
    let j = floor(random(i + 1));
    let temp = pool[i];
    pool[i] = pool[j];
    pool[j] = temp;
  }
  gridLetters = new Array(gridCols * gridRows).fill("");
  for (let i = 0; i < pool.length; i++) {
    gridLetters[i] = pool[i];
  }
  createStuff();
}

function cycleWordBank() {
  currentWordIdx = (currentWordIdx + 1) % wordBank.length;
  let target = wordBank[currentWordIdx];
  if (target === "FULL_SEQ") {
    setFullAlphabetSequential();
  } else if (target === "FULL_SHUF") {
    setFullAlphabetShuffled();
  } else {
    setWord(target);
  }
}

function draw() {
  let pal = getActivePalette();
  background(pal.bg[0], pal.bg[1], pal.bg[2]);

  // Calculate live breathing cycle
  if (currentMode === 'breathing') {
    pulseCycle = sin(frameCount * 0.018);
  } else if (currentMode === 'stable') {
    pulseCycle = 0.95; // Locked to high coherence
  } else if (currentMode === 'dissolve') {
    pulseCycle = -0.85; // Locked to high dispersion
  }

  drawTelemetryHUD();
  drawTechnicalGrid();
  drawPetriDishes();

  for (let i = 0; i < ballCollection.length; i++) {
    ballCollection[i].run();
  }  

  drawPetriDishGlassOverlays();

  theta += 0.15;
}

function drawTelemetryHUD() {
  let pal = getActivePalette();

  // Top instrumentation header background
  fill(max(0, pal.bg[0] - 5), max(0, pal.bg[1] - 5), max(0, pal.bg[2] - 5));
  noStroke();
  rect(0, 0, width, gridTop);

  stroke(pal.border[0], pal.border[1], pal.border[2], 120);
  strokeWeight(1);
  line(0, gridTop, width, gridTop);

  // Mode status readout (Right)
  fill(pal.hud[0], pal.hud[1], pal.hud[2], 220);
  noStroke();
  textFont("IBM Plex Mono, monospace");
  textSize(9.5);
  textAlign(RIGHT, CENTER);

  let stateLabel = "";
  if (currentMode === 'breathing') {
    stateLabel = pulseCycle > 0 
      ? `CYCLE: COALESCING [${floor(map(pulseCycle, 0, 1, 0, 100))}%]`
      : `CYCLE: BLOOMING [${floor(map(pulseCycle, 0, -1, 0, 100))}%]`;
  } else if (currentMode === 'stable') {
    stateLabel = "MODE: STABLE LOCK [1]";
  } else if (currentMode === 'dissolve') {
    stateLabel = "MODE: DISSOLVED BLOOM [2]";
  }

  text(stateLabel, width - 15, gridTop / 2);
  
  // Interactive control key hints (Left)
  fill(pal.hud[0], pal.hud[1], pal.hud[2], 140);
  textFont("Roboto Mono, monospace");
  textAlign(LEFT, CENTER);
  let containTag = isContained ? "[C] CONTAIN:ON" : "[C] CONTAIN:OFF";
  let orientTag = `[O] ${textOrientation}`;
  let themeTag = isCustomPaletteMode ? `[M] ${pal.name}` : `[M] MONO`;
  text(`[1-3] MODES  ${containTag}  ${orientTag}  ${themeTag}  [SPC] NEXT`, 15, gridTop / 2);
}

function drawTechnicalGrid() {
  let cellW = width / gridCols;
  let cellH = (height - gridTop) / gridRows;

  stroke(255, 10);
  strokeWeight(1);
  
  // Grid lines
  for (let c = 1; c < gridCols; c++) {
    line(c * cellW, gridTop, c * cellW, height);
  }
  for (let r = 1; r < gridRows; r++) {
    line(0, gridTop + r * cellH, width, gridTop + r * cellH);
  }

  // Corner crosshairs for each cell
  stroke(100, 180, 255, 40);
  for (let r = 0; r <= gridRows; r++) {
    for (let c = 0; c <= gridCols; c++) {
      let x = c * cellW;
      let y = gridTop + r * cellH;
      line(x - 4, y, x + 4, y);
      line(x, y - 4, x, y + 4);
    }
  }
}

function drawPetriDishes() {
  let pal = getActivePalette();
  let cellW = width / gridCols;
  let cellH = (height - gridTop) / gridRows;
  let dishRadius = min(cellW, cellH) * 0.42;
  let dishDiameter = dishRadius * 2;
  let fontSize = min(cellW, cellH) * 0.52;

  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      let cx = c * cellW + cellW / 2;
      let cy = gridTop + r * cellH + cellH / 2;

      // 1. Outer well shadow / recess
      noFill();
      stroke(0, 0, 0, 120);
      strokeWeight(3);
      circle(cx, cy + 1, dishDiameter + 6);

      // 2. Agar substrate base plate
      fill(pal.agar[0], pal.agar[1], pal.agar[2]);
      stroke(pal.border[0], pal.border[1], pal.border[2], 120);
      strokeWeight(1.2);
      circle(cx, cy, dishDiameter);

      // 3. Faint Ghost Glyph Underlay (only if cell has a letter)
      let char = gridLetters[(r * gridCols + c)];
      if (char && char !== "") {
        let ghostAlpha = max(0, map(pulseCycle, -1, 1, -2, 14));
        if (ghostAlpha > 0) {
          fill(pal.accent[0], pal.accent[1], pal.accent[2], ghostAlpha * 8);
          noStroke();
          textFont("Space Grotesk, Chivo Mono, sans-serif");
          textStyle(BOLD);
          textSize(fontSize);
          textAlign(CENTER, CENTER);
          text(char, cx, cy + fontSize * 0.03);
        }
      }

      // 4. Inner meniscus ring
      noFill();
      stroke(pal.accent[0], pal.accent[1], pal.accent[2], 35);
      strokeWeight(1);
      circle(cx, cy, dishDiameter * 0.93);

      // 5. Precision radial calibration ticks around dish rim
      stroke(pal.accent[0], pal.accent[1], pal.accent[2], 55);
      strokeWeight(0.8);
      let numTicks = 24;
      for (let t = 0; t < numTicks; t++) {
        let ang = (TWO_PI / numTicks) * t;
        let r1 = dishRadius;
        let isMajor = (t % 6 === 0);
        let r2 = isMajor ? dishRadius - 6 : dishRadius - 3;
        stroke(isMajor ? color(pal.glassRim[0], pal.glassRim[1], pal.glassRim[2], 120) : color(pal.accent[0], pal.accent[1], pal.accent[2], 45));
        line(cx + cos(ang) * r1, cy + sin(ang) * r1, cx + cos(ang) * r2, cy + sin(ang) * r2);
      }

      // 6. Well coordinate tag (A1, A2, A3...) - positioned inside the petri dish
      let rowLabel = String.fromCharCode(65 + r);
      let colLabel = (c + 1);
      fill(pal.hud[0], pal.hud[1], pal.hud[2], 190);
      noStroke();
      textFont("Chivo Mono, monospace");
      let tagSize = constrain(dishRadius * 0.15, 6.5, 8.5);
      textSize(tagSize);
      textAlign(LEFT, TOP);
      text(rowLabel + colLabel, cx - dishRadius * 0.52, cy - dishRadius * 0.56);
    }
  }
}

function drawPetriDishGlassOverlays() {
  let pal = getActivePalette();
  let cellW = width / gridCols;
  let cellH = (height - gridTop) / gridRows;
  let dishRadius = min(cellW, cellH) * 0.42;
  let dishDiameter = dishRadius * 2;

  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      let cx = c * cellW + cellW / 2;
      let cy = gridTop + r * cellH + cellH / 2;

      noFill();

      // Outer glass wall lip bevel
      stroke(pal.glassRim[0], pal.glassRim[1], pal.glassRim[2], 80);
      strokeWeight(1.5);
      circle(cx, cy, dishDiameter + 4);

      // Primary glass rim
      stroke(pal.glassRim[0], pal.glassRim[1], pal.glassRim[2], 150);
      strokeWeight(1);
      circle(cx, cy, dishDiameter);

      // Upper-left bright specular reflection arc
      stroke(255, 255, 255, 220);
      strokeWeight(1.8);
      arc(cx, cy, dishDiameter + 2, dishDiameter + 2, -PI * 0.85, -PI * 0.35);

      // Inner glass refraction arc
      stroke(pal.glassRim[0], pal.glassRim[1], pal.glassRim[2], 110);
      strokeWeight(1);
      arc(cx, cy, dishDiameter - 3, dishDiameter - 3, -PI * 0.8, -PI * 0.4);

      // Bottom-right ambient bounce light reflection
      stroke(pal.accent[0], pal.accent[1], pal.accent[2], 75);
      strokeWeight(1);
      arc(cx, cy, dishDiameter + 2, dishDiameter + 2, PI * 0.15, PI * 0.55);
    }
  }
}

function keyPressed() {
  if (key === '1') {
    currentMode = 'stable';
  } else if (key === '2') {
    currentMode = 'dissolve';
  } else if (key === '3') {
    currentMode = 'breathing';
  } else if (key === 'c' || key === 'C') {
    isContained = !isContained;
  } else if (key === 'o' || key === 'O') {
    toggleOrientation();
  } else if (key === 'm' || key === 'M') {
    isCustomPaletteMode = !isCustomPaletteMode;
  } else if (key === 'r' || key === 'R') {
    setFullAlphabetShuffled();
  } else if (key === ' ') {
    cycleWordBank();
  } else if (keyCode === BACKSPACE || keyCode === DELETE) {
    if (activeWord.length > 0) {
      setWord(activeWord.substring(0, activeWord.length - 1));
    }
  } else if (key.length === 1 && key.toUpperCase() >= 'A' && key.toUpperCase() <= 'Z') {
    let char = key.toUpperCase();
    if (activeWord.length < 16) {
      setWord(activeWord + char);
    }
  }

  // Prevent default browser scrolling when pressing Spacebar / Backspace
  return false;
}

function getLuminance(x, y) {
  let ix = constrain(floor(x), 0, width - 1);
  let iy = constrain(floor(y), 0, height - 1);
  return letterGraphics.pixels[(ix + iy * width) * 4];
}

function createStuff() {
  ballCollection = [];
  
  letterGraphics.background(255);
  letterGraphics.noStroke();
  letterGraphics.fill(0);
  letterGraphics.textFont("Space Grotesk, Chivo Mono, sans-serif");
  letterGraphics.textStyle(BOLD);
  
  let cellW = width / gridCols;
  let cellH = (height - gridTop) / gridRows;
  let fontSize = min(cellW, cellH) * 0.52;
  letterGraphics.textSize(fontSize);
  letterGraphics.textAlign(CENTER, CENTER);
  
  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      let x = c * cellW + cellW / 2;
      let y = gridTop + r * cellH + cellH / 2 + fontSize * 0.03;
      let char = gridLetters[r * gridCols + c];
      if (char && char !== "") {
        letterGraphics.text(char, x, y);
      }
    }
  }
  
  // Reduced gaussian blur (2.0) to preserve sharp, crisp letter anatomy & inner holes
  letterGraphics.filter(BLUR, 2);
  letterGraphics.loadPixels();

  // Clip letterGraphics outside dish bounds so no text spills into dish margin
  let cellRadius = min(cellW, cellH) * 0.40;
  for (let y = floor(gridTop); y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = floor((y - gridTop) / cellH);
      let c = floor(x / cellW);
      if (r >= 0 && r < gridRows && c >= 0 && c < gridCols) {
        let cx = c * cellW + cellW / 2;
        let cy = gridTop + r * cellH + cellH / 2;
        if (dist(x, y, cx, cy) > cellRadius) {
          let i = (x + y * width) * 4;
          letterGraphics.pixels[i] = 255;
          letterGraphics.pixels[i + 1] = 255;
          letterGraphics.pixels[i + 2] = 255;
        }
      }
    }
  }

  // Seed particles with probability proportional to darkness (letter density)
  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      let char = gridLetters[r * gridCols + c];
      if (!char || char === "") continue; // Skip empty dishes!

      let cx = c * cellW + cellW / 2;
      let cy = gridTop + r * cellH + cellH / 2;
      
      let count = 0;
      let attempts = 0;
      while (count < numPerCell && attempts < numPerCell * 40) {
        let ang = random(TWO_PI);
        let distR = sqrt(random()) * cellRadius;
        let x = cx + cos(ang) * distR;
        let y = cy + sin(ang) * distR;
        
        let lum = getLuminance(x, y);
        let acceptProb = map(lum, 0, 255, 0.98, 0.02);
        if (random() < acceptProb) {
          let org = createVector(x, y);
          let radius = random(1.5, 4.0);
          let loc = createVector(org.x + radius, org.y);
          let offSet = random(TWO_PI);
          let dir = random(1) > 0.5 ? 1 : -1;
          
          let myBall = new Ball(org, loc, radius, dir, offSet, cx, cy, cellRadius);
          ballCollection.push(myBall);
          count++;
        }
        attempts++;
      }
    }
  }
}

class Ball {
  constructor(_org, _loc, _radius, _dir, _offSet, _cx, _cy, _dishR) {
    this.home = _org.copy(); // Permanent anchor point on the letter glyph
    this.org = _org.copy();
    this.loc = _loc;
    this.radius = _radius;
    this.dir = _dir;
    this.offSet = _offSet;
    this.cx = _cx;
    this.cy = _cy;
    this.dishR = _dishR;
    
    this.sz = random(1.5, 2.5);
    this.d = 16;
    this.luminance = 128;
  }

  run() {
    this.move();
    this.display();
  }

  move() {
    let step = 3;

    // 1. Move home anchor along letter contour/spine
    let homeLum = getLuminance(this.home.x, this.home.y);
    let hLeft = getLuminance(this.home.x - step, this.home.y);
    let hRight = getLuminance(this.home.x + step, this.home.y);
    let hUp = getLuminance(this.home.x, this.home.y - step);
    let hDown = getLuminance(this.home.x, this.home.y + step);
    let hGradX = (hRight - hLeft) / (2 * step);
    let hGradY = (hDown - hUp) / (2 * step);

    // Tangential circulation along stroke + pull to stroke center
    let tangX = -hGradY * 0.3 * this.dir;
    let tangY = hGradX * 0.3 * this.dir;
    let hPull = (homeLum > 60) ? 0.12 : 0.02;
    this.home.x += tangX - hGradX * hPull;
    this.home.y += tangY - hGradY * hPull;

    // 2. Elastic spring tether from org to home anchor
    let toHomeX = this.home.x - this.org.x;
    let toHomeY = this.home.y - this.org.y;
    // In Dissolve Mode (pulseCycle <= 0), spring drops to 0, completely breaking away from the form!
    let springStrength = max(0, map(pulseCycle, -0.2, 1, 0.0, 0.32));
    
    // Organic noise wandering flow (surges to 3.5x in Dissolve Mode)
    let angle = noise(this.org.x * 0.015, this.org.y * 0.015, frameCount * 0.01) * TWO_PI * 3;
    let noiseAmp = map(pulseCycle, -1, 1, 3.5, 0.4);
    let flowX = cos(angle) * noiseAmp;
    let flowY = sin(angle) * noiseAmp;

    this.org.x += toHomeX * springStrength + flowX;
    this.org.y += toHomeY * springStrength + flowY;

    if (isContained) {
      // Hard clamp home anchor inside dish
      let maxHomeR = this.dishR - 5;
      let dHome = dist(this.home.x, this.home.y, this.cx, this.cy);
      if (dHome > maxHomeR) {
        let ang = atan2(this.home.y - this.cy, this.home.x - this.cx);
        this.home.x = this.cx + cos(ang) * maxHomeR;
        this.home.y = this.cy + sin(ang) * maxHomeR;
      }

      // Hard clamp org inside dish
      let maxOrgR = this.dishR - 4;
      let dOrg = dist(this.org.x, this.org.y, this.cx, this.cy);
      if (dOrg > maxOrgR) {
        let ang = atan2(this.org.y - this.cy, this.org.x - this.cx);
        this.org.x = this.cx + cos(ang) * maxOrgR;
        this.org.y = this.cy + sin(ang) * maxOrgR;
      }
    } else {
      // Wrap around canvas boundaries when free roaming
      this.org.x = (this.org.x + width) % width;
      if (this.org.y < gridTop) this.org.y = height - 10;
      if (this.org.y >= height) this.org.y = gridTop + 10;
    }

    this.luminance = getLuminance(this.org.x, this.org.y);

    // Organic undulating movement
    let wobbleR = this.radius * (0.8 + 0.4 * sin(theta * 0.5 + this.offSet));
    this.loc.x = this.org.x + sin(theta * this.dir + this.offSet) * wobbleR;
    this.loc.y = this.org.y + cos(theta * this.dir + this.offSet) * wobbleR;

    // Hard clamp drawn particle location (loc) inside glass rim
    if (isContained) {
      let maxLocR = this.dishR - 3;
      let dLoc = dist(this.loc.x, this.loc.y, this.cx, this.cy);
      if (dLoc > maxLocR) {
        let ang = atan2(this.loc.y - this.cy, this.loc.x - this.cx);
        this.loc.x = this.cx + cos(ang) * maxLocR;
        this.loc.y = this.cy + sin(ang) * maxLocR;
      }
    }
  }

  lineBetween() {
    let connections = 0;
    let maxConnections = 2; // Capping connections prevents rigid triangular meshes

    // Dynamically adjust reach during bloom vs crystallization
    this.d = map(pulseCycle, -1, 1, 22, 15);

    // Letter intensity factor (1 in core, 0.2 in negative space)
    let intensity = map(this.luminance, 0, 255, 1.0, 0.25);

    for (let i = 0; i < ballCollection.length; i++) {
      if (connections >= maxConnections) break;
      let other = ballCollection[i];
      if (other === this) continue;
      
      let dX = this.loc.x - other.loc.x;
      let dY = this.loc.y - other.loc.y;
      let distSq = dX * dX + dY * dY;
      
      if (distSq > 0 && distSq < this.d * this.d) {
        let distance = sqrt(distSq);
        let a = map(distance, 0, this.d, 190, 15) * intensity;
        
        // Organic curved tendril / filament line
        stroke(175, 220, 255, a);
        strokeWeight(map(distance, 0, this.d, 1.3, 0.3) * intensity);
        
        // Midpoint noise wobble for organic irregular filaments
        let midX = (this.loc.x + other.loc.x) * 0.5;
        let midY = (this.loc.y + other.loc.y) * 0.5;
        let wobble = (noise(midX * 0.05, midY * 0.05, frameCount * 0.02) - 0.5) * 7;
        let perpX = -(other.loc.y - this.loc.y) / distance * wobble;
        let perpY = (other.loc.x - this.loc.x) / distance * wobble;
        
        noFill();
        beginShape();
        vertex(this.loc.x, this.loc.y);
        quadraticVertex(midX + perpX, midY + perpY, other.loc.x, other.loc.y);
        endShape();
        
        connections++;
      }
    }
  }

  display() {
    let pal = getActivePalette();
    noStroke();
    let intensity = map(this.luminance, 0, 255, 1.0, 0.3);
    
    // Soft luminous spore glow & nucleus fading out gently in negative space
    fill(pal.nodeOuter[0], pal.nodeOuter[1], pal.nodeOuter[2], 55 * intensity);
    circle(this.loc.x, this.loc.y, this.sz * 2.5);
    fill(pal.nodeCore[0], pal.nodeCore[1], pal.nodeCore[2], 220 * intensity);
    circle(this.loc.x, this.loc.y, this.sz);
  }
}
