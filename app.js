const audio = document.querySelector('#audio');
const ui = {
  tabs: document.querySelector('#example-tabs'),
  title: document.querySelector('#example-title'),
  instrument: document.querySelector('#example-instrument'),
  request: document.querySelector('#request-label'),
  midi: document.querySelector('#midi-download'),
  image: document.querySelector('#spectrogram-image'),
  mask: document.querySelector('#mask-overlay'),
  maskDimmer: document.querySelector('#mask-dimmer'),
  maskToggle: document.querySelector('#mask-toggle'),
  spectrogram: document.querySelector('#spectrogram'),
  selection: document.querySelector('#selection'),
  playhead: document.querySelector('#playhead'),
  channelLabel: document.querySelector('#channel-label'),
  currentTime: document.querySelector('#current-time'),
  duration: document.querySelector('#duration'),
  primaryChannels: document.querySelector('#primary-channels'),
  baselineChannels: document.querySelector('#baseline-channels'),
  playWhole: document.querySelector('#play-whole'),
  playRegion: document.querySelector('#play-region'),
  stop: document.querySelector('#stop'),
  region: document.querySelector('#region-readout'),
  roll: document.querySelector('#piano-roll'),
  scoreNote: document.querySelector('#score-note'),
  figure: document.querySelector('#figure-grid'),
  showFigureExample: document.querySelector('#show-figure-example'),
};

const state = {
  manifest: null,
  example: null,
  channel: null,
  start: 0,
  end: 0,
  selectedEvent: null,
  segmentEnd: null,
  dragStart: null,
  animation: null,
  loadToken: 0,
};

const channelNotes = {
  mixture: 'Full polyphonic recording',
  target: 'Ground-truth isolated note',
  aso: 'Strongest finalized separator output',
  midi: 'Score rendered as a simple reference synth',
  nmf: 'Score-informed signal-processing baseline',
  hpss: 'Dual-branch neural baseline',
  symmetric: 'Separator output before joint ASO allocation',
};

function pitchName(pitch) {
  const names = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'];
  return `${names[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function currentChannel(id = state.channel) {
  const channel = state.example.channels.find((item) => item.id === id);
  const note = state.example.notes.find((item) => item.event === state.selectedEvent);
  return note?.outputs?.[id] ? {...channel, ...note.outputs[id]} : channel;
}

function setSelection(start, end, event = null, audition = false) {
  const duration = state.example.duration;
  const queryChanged = event !== null && event !== state.selectedEvent;
  state.start = clamp(Math.min(start, end), 0, duration);
  state.end = clamp(Math.max(start, end), 0, duration);
  if (state.end - state.start < 0.04) state.end = Math.min(duration, state.start + 0.12);
  if (event !== null) state.selectedEvent = event;
  const left = state.start / duration * 100;
  const width = (state.end - state.start) / duration * 100;
  ui.selection.style.left = `${left}%`;
  ui.selection.style.width = `${width}%`;
  ui.region.textContent = `Region ${state.start.toFixed(2)}–${state.end.toFixed(2)} s`;
  document.querySelectorAll('.roll-note').forEach((note) => {
    note.classList.toggle('roll-note--selected', Number(note.dataset.event) === state.selectedEvent);
  });
  const selected = state.example.notes.find((note) => note.event === state.selectedEvent);
  ui.mask.src = selected?.outputs?.aso?.mask ? `${selected.outputs.aso.mask}?v=2` : '';
  const hideMask = !ui.maskToggle.checked || !ui.mask.src;
  ui.mask.classList.toggle('mask-overlay--hidden', hideMask);
  ui.maskDimmer.classList.toggle('mask-dimmer--hidden', hideMask);
  if (selected) {
    ui.request.textContent = `Frozen request ${state.example.pieceId}:${String(selected.event).padStart(4, '0')} · queried pitch ${pitchName(selected.pitch)}`;
  }
  ui.scoreNote.textContent = selected
    ? `${pitchName(selected.pitch)}, ${state.start.toFixed(2)}–${state.end.toFixed(2)} s. This note is now the separator query.`
    : 'The selected region can be compared across every audio output.';
  if (audition) {
    selectChannel('aso', () => play(state.start, state.end));
  } else if (queryChanged && !['mixture', 'midi'].includes(state.channel)) {
    selectChannel(state.channel);
  }
}

function renderTabs() {
  ui.tabs.replaceChildren(...state.manifest.examples.map((example) => {
    const button = document.createElement('button');
    button.className = 'tab';
    button.type = 'button';
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-selected', String(example.id === state.example.id));
    button.textContent = `${example.instrument} · ${example.title}`;
    button.addEventListener('click', () => selectExample(example.id));
    return button;
  }));
}

function channelButton(channel) {
  const button = document.createElement('button');
  button.className = 'source-button';
  button.type = 'button';
  button.dataset.channel = channel.id;
  button.setAttribute('aria-pressed', String(channel.id === state.channel));
  const strong = document.createElement('strong');
  strong.textContent = channel.label;
  const small = document.createElement('small');
  small.textContent = channelNotes[channel.id] || '';
  button.append(strong, small);
  button.addEventListener('click', () => selectChannel(channel.id));
  return button;
}

function renderChannels() {
  const primary = ['mixture', 'aso', 'midi'];
  ui.primaryChannels.replaceChildren(...state.example.channels.filter((c) => primary.includes(c.id)).map(channelButton));
  ui.baselineChannels.replaceChildren(...state.example.channels.filter((c) => !primary.includes(c.id)).map(channelButton));
}

function renderScore() {
  const notes = state.example.notes;
  const low = Math.min(...notes.map((note) => note.pitch)) - 1;
  const high = Math.max(...notes.map((note) => note.pitch)) + 1;
  const rows = high - low + 1;
  const width = 1000;
  const height = 210;
  const labelWidth = 38;
  const plotWidth = width - labelWidth;
  const rowHeight = height / rows;
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `Piano-roll score for ${state.example.title}`);

  for (let pitch = low; pitch <= high; pitch += 1) {
    const y = (high - pitch) * rowHeight;
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', labelWidth);
    line.setAttribute('x2', width);
    line.setAttribute('y1', y);
    line.setAttribute('y2', y);
    line.setAttribute('class', 'roll-grid');
    svg.append(line);
    if (pitch % 12 === 0 || pitch === low || pitch === high) {
      const label = document.createElementNS(ns, 'text');
      label.setAttribute('x', 4);
      label.setAttribute('y', y + rowHeight * 0.72);
      label.setAttribute('class', 'roll-label');
      label.textContent = pitchName(pitch);
      svg.append(label);
    }
  }

  notes.forEach((note) => {
    const rect = document.createElementNS(ns, 'rect');
    const x = labelWidth + note.start / state.example.duration * plotWidth;
    const noteWidth = Math.max(5, (note.end - note.start) / state.example.duration * plotWidth);
    const y = (high - note.pitch) * rowHeight + 1;
    rect.setAttribute('x', x);
    rect.setAttribute('y', y);
    rect.setAttribute('width', noteWidth);
    rect.setAttribute('height', Math.max(4, rowHeight - 2));
    rect.setAttribute('rx', 2);
    rect.setAttribute('tabindex', '0');
    rect.setAttribute('role', 'button');
    rect.setAttribute('aria-label', `${pitchName(note.pitch)}, ${note.start.toFixed(2)} to ${note.end.toFixed(2)} seconds${note.target ? ', queried note' : ''}`);
    rect.dataset.event = note.event;
    rect.setAttribute('class', 'roll-note');
    const choose = () => setSelection(note.start, note.end, note.event, true);
    rect.addEventListener('click', choose);
    rect.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); }
    });
    svg.append(rect);
  });

  const playhead = document.createElementNS(ns, 'line');
  playhead.setAttribute('x1', labelWidth);
  playhead.setAttribute('x2', labelWidth);
  playhead.setAttribute('y1', 0);
  playhead.setAttribute('y2', height);
  playhead.setAttribute('class', 'roll-playhead');
  playhead.id = 'roll-playhead';
  svg.append(playhead);
  ui.roll.replaceChildren(svg);
}

function selectChannel(id, onReady = null) {
  const channel = currentChannel(id);
  if (!channel) return;
  const wasPlaying = !audio.paused;
  const position = audio.currentTime || 0;
  audio.pause();
  const loadToken = ++state.loadToken;
  state.channel = id;
  audio.src = channel.audio;
  audio.addEventListener('loadedmetadata', () => {
    if (loadToken !== state.loadToken) return;
    audio.currentTime = clamp(position, 0, state.example.duration);
    if (onReady) onReady();
    else if (wasPlaying) audio.play().catch(() => {});
  }, {once: true});
  audio.load();
  ui.image.src = channel.spectrogram;
  ui.channelLabel.textContent = channel.label;
  document.querySelectorAll('.source-button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.channel === id));
  });
}

function selectExample(id) {
  audio.pause();
  state.segmentEnd = null;
  state.example = state.manifest.examples.find((example) => example.id === id);
  state.channel = 'mixture';
  state.selectedEvent = state.example.targetEvent;
  ui.title.textContent = state.example.title;
  ui.instrument.textContent = state.example.instrument;
  ui.request.textContent = `Frozen request ${state.example.requestId} · queried pitch ${pitchName(state.example.targetPitch)}`;
  ui.duration.textContent = state.example.duration.toFixed(2);
  ui.midi.href = state.example.midi;
  ui.midi.download = `${state.example.id}-score.mid`;
  audio.src = currentChannel().audio;
  audio.load();
  ui.image.src = currentChannel().spectrogram;
  ui.channelLabel.textContent = currentChannel().label;
  renderTabs();
  renderChannels();
  renderScore();
  const target = state.example.notes.find((note) => note.target);
  setSelection(target.start, target.end, target.event);
}

function play(start, end = null) {
  audio.currentTime = clamp(start, 0, state.example.duration);
  state.segmentEnd = end;
  audio.play().catch((error) => console.warn('Playback was blocked:', error));
}

function stop() {
  audio.pause();
  state.segmentEnd = null;
}

function updatePlayhead() {
  if (!state.example) return;
  if (state.segmentEnd !== null && audio.currentTime >= state.segmentEnd) stop();
  const ratio = clamp((audio.currentTime || 0) / state.example.duration, 0, 1);
  ui.playhead.style.left = `${ratio * 100}%`;
  ui.currentTime.textContent = (audio.currentTime || 0).toFixed(2);
  const rollHead = document.querySelector('#roll-playhead');
  if (rollHead) {
    const x = 38 + ratio * (1000 - 38);
    rollHead.setAttribute('x1', x);
    rollHead.setAttribute('x2', x);
  }
  state.animation = requestAnimationFrame(updatePlayhead);
}

function spectrogramTime(event) {
  const bounds = ui.spectrogram.getBoundingClientRect();
  return clamp((event.clientX - bounds.left) / bounds.width, 0, 1) * state.example.duration;
}

function renderFigureCompanion() {
  const guitar = state.manifest.examples.find((example) => example.id === 'guitar');
  const figureOrder = ['mixture', 'target', 'nmf', 'hpss', 'symmetric', 'aso'];
  const cards = figureOrder.map((id) => guitar.channels.find((channel) => channel.id === id)).map((channel) => {
    const button = document.createElement('button');
    button.className = 'figure-card';
    button.type = 'button';
    const image = document.createElement('img');
    image.src = channel.spectrogram;
    image.alt = '';
    const label = document.createElement('span');
    label.textContent = channel.label;
    const detail = document.createElement('small');
    detail.textContent = 'Load and listen';
    label.append(detail);
    button.append(image, label);
    button.addEventListener('click', () => {
      selectExample('guitar');
      selectChannel(channel.id);
      document.querySelector('.demo-shell').scrollIntoView({behavior: 'smooth'});
      play(0);
    });
    return button;
  });
  ui.figure.replaceChildren(...cards);
}

ui.playWhole.addEventListener('click', () => play(0));
ui.playRegion.addEventListener('click', () => play(state.start, state.end));
ui.stop.addEventListener('click', stop);
ui.showFigureExample.addEventListener('click', () => {
  selectExample('guitar');
  document.querySelector('.demo-shell').scrollIntoView({behavior: 'smooth'});
});

ui.maskToggle.addEventListener('change', () => {
  const hideMask = !ui.maskToggle.checked || !ui.mask.src;
  ui.mask.classList.toggle('mask-overlay--hidden', hideMask);
  ui.maskDimmer.classList.toggle('mask-dimmer--hidden', hideMask);
});

ui.spectrogram.addEventListener('pointerdown', (event) => {
  state.dragStart = spectrogramTime(event);
  ui.spectrogram.setPointerCapture(event.pointerId);
  setSelection(state.dragStart, state.dragStart + 0.04);
});
ui.spectrogram.addEventListener('pointermove', (event) => {
  if (state.dragStart !== null) setSelection(state.dragStart, spectrogramTime(event));
});
ui.spectrogram.addEventListener('pointerup', (event) => {
  if (state.dragStart !== null) setSelection(state.dragStart, spectrogramTime(event));
  state.dragStart = null;
});

fetch('assets/manifest.json')
  .then((response) => {
    if (!response.ok) throw new Error(`Could not load demo manifest (${response.status})`);
    return response.json();
  })
  .then((manifest) => {
    state.manifest = manifest;
    selectExample(manifest.examples[0].id);
    renderFigureCompanion();
    updatePlayhead();
  })
  .catch((error) => {
    ui.title.textContent = 'The demo assets could not be loaded.';
    ui.request.textContent = 'Serve this folder through a web server or GitHub Pages.';
    console.error(error);
  });
