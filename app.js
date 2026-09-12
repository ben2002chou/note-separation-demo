const audio = document.querySelector('#audio');
const auditionAudio = document.querySelector('#note-audition');
const ui = {
  tabs: document.querySelector('#example-tabs'),
  title: document.querySelector('#example-title'),
  instrument: document.querySelector('#example-instrument'),
  request: document.querySelector('#request-label'),
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
  binCandidates: document.querySelector('#bin-candidates'),
  playWhole: document.querySelector('#play-whole'),
  playRegion: document.querySelector('#play-region'),
  stop: document.querySelector('#stop'),
  region: document.querySelector('#region-readout'),
  roll: document.querySelector('#piano-roll'),
  scoreNote: document.querySelector('#score-note'),
};

const state = {
  manifest: null,
  example: null,
  channel: null,
  start: 0,
  end: 0,
  selectedEvent: null,
  segmentEnd: null,
  pointerStart: null,
  dragging: false,
  animation: null,
  loadToken: 0,
  auditionOffset: null,
  hitMapImage: null,
  hitMapCanvas: document.createElement('canvas'),
};

const channelNotes = {
  mixture: 'Full polyphonic recording',
  target: 'Ground-truth isolated note',
  score_informed_nmf: 'Score-informed harmonic NMF baseline',
  aso: 'Final jointly allocated note output',
  independent_ungated: 'Independent branches without the selective octave gate',
  independent_selective: 'Independent branches with the selective octave gate',
  symmetric_ungated: 'Bidirectional branch interaction without the selective gate',
  symmetric_selective: 'Bidirectional branch interaction with the selective gate',
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

function mixtureChannel() {
  return state.example.channels.find((item) => item.id === 'mixture');
}

function setSelection(start, end, event = null, refreshChannel = true) {
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
  document.querySelectorAll('.roll-waveform').forEach((waveform) => {
    waveform.classList.toggle('roll-waveform--selected', Number(waveform.dataset.event) === state.selectedEvent);
  });
  const selected = state.example.notes.find((note) => note.event === state.selectedEvent);
  ui.mask.src = selected?.outputs?.aso?.mask ? `${selected.outputs.aso.mask}?v=20260912-0300` : '';
  const hideMask = !ui.maskToggle.checked || !ui.mask.src;
  ui.mask.classList.toggle('mask-overlay--hidden', hideMask);
  ui.maskDimmer.classList.toggle('mask-dimmer--hidden', hideMask);
  if (selected) {
    ui.request.textContent = state.example.qualitative
      ? `${state.example.requestLabel} · queried pitch ${pitchName(selected.pitch)}`
      : `Frozen request ${state.example.pieceId}:${String(selected.event).padStart(4, '0')} · queried pitch ${pitchName(selected.pitch)}`;
  }
  ui.scoreNote.textContent = selected
    ? `${pitchName(selected.pitch)}, ${state.start.toFixed(2)}–${state.end.toFixed(2)} s · ${currentChannel()?.label || 'selected method'}`
    : 'The selected region can be compared across every audio output.';
  if (queryChanged && refreshChannel && state.channel !== 'mixture') {
    selectChannel(state.channel);
  }
}

function auditionNote(note, channelId = state.channel) {
  ui.binCandidates.hidden = true;
  stop();
  state.channel = channelId;
  setSelection(note.start, note.end, note.event, false);
  selectChannel(channelId);
  const audition = note.auditions?.[channelId];
  if (!audition) return;
  auditionAudio.src = audition.audio;
  auditionAudio.currentTime = 0;
  state.auditionOffset = audition.start || 0;
  auditionAudio.play().catch((error) => console.warn('Note audition was blocked:', error));
}

function loadHitMap() {
  const exampleId = state.example.id;
  const image = new Image();
  state.hitMapImage = image;
  image.addEventListener('load', () => {
    if (state.example.id !== exampleId || state.hitMapImage !== image) return;
    const canvas = state.hitMapCanvas;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext('2d', {willReadFrequently: true}).drawImage(image, 0, 0);
  }, {once: true});
  image.src = `${state.example.hitMap}?v=20260912-0300`;
}

function fallbackSpectralNote(time, frequency) {
  const active = state.example.notes.filter((note) => time >= note.start - 0.04 && time <= note.end + 0.25);
  if (!active.length) return null;
  const candidates = active;
  const safeFrequency = Math.max(20, frequency);
  return [...candidates].sort((a, b) => {
    const harmonicError = (note) => {
      const fundamental = 440 * 2 ** ((note.pitch - 69) / 12);
      const harmonic = clamp(Math.round(safeFrequency / fundamental), 1, 20);
      return Math.abs(1200 * Math.log2(safeFrequency / (fundamental * harmonic)));
    };
    return harmonicError(a) - harmonicError(b)
      || Math.abs(a.start - time) - Math.abs(b.start - time);
  })[0];
}

function clearSpectralSelection(time, frequency) {
  stop();
  state.selectedEvent = null;
  ui.mask.src = '';
  ui.mask.classList.add('mask-overlay--hidden');
  ui.maskDimmer.classList.add('mask-dimmer--hidden');
  document.querySelectorAll('.roll-note, .roll-waveform').forEach((note) => {
    note.classList.remove('roll-note--selected', 'roll-waveform--selected');
  });
  ui.request.textContent = `No allocated note at ${time.toFixed(2)} s, ${frequency >= 1000 ? `${(frequency / 1000).toFixed(1)} kHz` : `${Math.round(frequency)} Hz`}`;
  ui.scoreNote.textContent = 'This time–frequency region is silent or has no confident note owner.';
  ui.binCandidates.hidden = false;
  ui.binCandidates.replaceChildren(document.createTextNode('No note energy was detected at this point.'));
}

function renderBinCandidates(notes, time, frequency, shared) {
  ui.binCandidates.hidden = false;
  const description = document.createElement('span');
  description.textContent = `${shared ? 'Shared energy' : 'Likely owner'} at ${time.toFixed(2)} s, ${frequency >= 1000 ? `${(frequency / 1000).toFixed(1)} kHz` : `${Math.round(frequency)} Hz`}:`;
  const buttons = notes.map((note, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'candidate-button';
    button.textContent = `${pitchName(note.pitch)}${index === 0 ? ' · strongest' : ' · alternate'}`;
    button.addEventListener('click', () => {
      auditionNote(note);
      renderBinCandidates(notes, time, frequency, shared);
    });
    return button;
  });
  ui.binCandidates.replaceChildren(description, ...buttons);
}

function selectSpectrogramPoint(event) {
  const bounds = ui.spectrogram.getBoundingClientRect();
  const xRatio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
  const yRatio = clamp((event.clientY - bounds.top) / bounds.height, 0, 1);
  const time = xRatio * state.example.duration;
  const frequency = (1 - yRatio) * 8000;
  let candidates = [];
  let shared = false;
  let ownershipMapReady = false;

  const image = state.hitMapImage;
  if (image?.complete && image.naturalWidth) {
    ownershipMapReady = true;
    const x = clamp(Math.floor(xRatio * image.naturalWidth), 0, image.naturalWidth - 1);
    const y = clamp(Math.floor(yRatio * image.naturalHeight), 0, image.naturalHeight - 1);
    const [winnerEvent, runnerEvent, packed] = state.hitMapCanvas
      .getContext('2d', {willReadFrequently: true})
      .getImageData(x, y, 1, 1).data;
    const strength = ((packed >> 4) & 15) * 17;
    const ambiguity = (packed & 15) * 17;
    if (strength >= 24) {
      const winner = state.example.notes.find((note) => note.event === winnerEvent);
      const runner = state.example.notes.find((note) => note.event === runnerEvent);
      if (winner) candidates.push(winner);
      shared = ambiguity >= 145 && runner && runner.event !== winner?.event;
      if (shared) candidates.push(runner);
    }
  }

  if (!candidates.length && !ownershipMapReady) {
    candidates = [fallbackSpectralNote(time, frequency)].filter(Boolean);
  }
  if (!candidates.length) {
    clearSpectralSelection(time, frequency);
    return;
  }
  auditionNote(candidates[0]);
  renderBinCandidates(candidates, time, frequency, shared);
}

function temporalDistance(a, b) {
  if (a.end < b.start) return b.start - a.end;
  if (b.end < a.start) return a.start - b.end;
  return 0;
}

function neighboringNote(note, key) {
  if (key === 'ArrowLeft' || key === 'ArrowRight') {
    const direction = key === 'ArrowRight' ? 1 : -1;
    const chordTolerance = 0.075;
    const candidates = state.example.notes.filter(
      (candidate) => direction * (candidate.start - note.start) > chordTolerance,
    );
    if (!candidates.length) return note;

    const nearestOnsetDistance = Math.min(
      ...candidates.map((candidate) => Math.abs(candidate.start - note.start)),
    );
    const onsetGroup = candidates.filter(
      (candidate) => Math.abs(Math.abs(candidate.start - note.start) - nearestOnsetDistance) <= chordTolerance,
    );
    onsetGroup.sort((a, b) => (
      Math.abs(a.pitch - note.pitch) - Math.abs(b.pitch - note.pitch)
      || temporalDistance(note, a) - temporalDistance(note, b)
      || a.event - b.event
    ));
    return onsetGroup[0];
  }

  const direction = key === 'ArrowUp' ? 1 : -1;
  const candidates = state.example.notes.filter(
    (candidate) => direction * (candidate.pitch - note.pitch) > 0,
  );
  candidates.sort((a, b) => (
    temporalDistance(note, a) - temporalDistance(note, b)
    || Math.abs(a.pitch - note.pitch) - Math.abs(b.pitch - note.pitch)
    || Math.abs(a.start - note.start) - Math.abs(b.start - note.start)
  ));
  return candidates[0] || note;
}

function navigateScore(note, key) {
  const next = neighboringNote(note, key);
  auditionNote(next);
  requestAnimationFrame(() => {
    ui.roll.querySelector(`[data-event="${next.event}"]`)?.focus();
  });
}

function renderTabs() {
  const publishedExamples = state.manifest.examples.filter((example) => example.id !== 'guitar');
  ui.tabs.replaceChildren(...publishedExamples.map((example) => {
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
  if (channel.id === 'aso') button.classList.add('source-button--ours');
  button.setAttribute('aria-pressed', String(channel.id === state.channel));
  const strong = document.createElement('strong');
  strong.textContent = channel.id === 'aso' ? 'Symmetric Gated + ASO · ours' : channel.label;
  const small = document.createElement('small');
  small.textContent = channelNotes[channel.id] || '';
  button.append(strong, small);
  button.addEventListener('click', () => {
    const selected = state.example.notes.find((note) => note.event === state.selectedEvent);
    if (selected) auditionNote(selected, channel.id);
    else selectChannel(channel.id);
  });
  return button;
}

function renderChannels() {
  const order = [
    'aso', 'symmetric_selective', 'symmetric_ungated',
    'independent_selective', 'independent_ungated', 'score_informed_nmf', 'target', 'mixture',
  ];
  const channels = order
    .map((id) => state.example.channels.find((channel) => channel.id === id))
    .filter(Boolean);
  ui.primaryChannels.replaceChildren(...channels.map(channelButton));
}

function renderScore() {
  const notes = state.example.notes;
  const low = Math.min(...notes.map((note) => note.pitch)) - 1;
  const high = Math.max(...notes.map((note) => note.pitch)) + 1;
  const rows = high - low + 1;
  const width = 1000;
  const height = 250;
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
    const row = document.createElementNS(ns, 'rect');
    row.setAttribute('x', labelWidth);
    row.setAttribute('y', y);
    row.setAttribute('width', plotWidth);
    row.setAttribute('height', rowHeight);
    row.setAttribute('class', [1, 3, 6, 8, 10].includes(pitch % 12) ? 'roll-row roll-row--black' : 'roll-row');
    svg.append(row);
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', labelWidth);
    line.setAttribute('x2', width);
    line.setAttribute('y1', y);
    line.setAttribute('y2', y);
    line.setAttribute('class', 'roll-grid');
    svg.append(line);
    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', 4);
    label.setAttribute('y', y + rowHeight * 0.72);
    label.setAttribute('class', 'roll-label');
    label.textContent = pitchName(pitch);
    svg.append(label);
  }

  for (let index = 0; index <= 8; index += 1) {
    const x = labelWidth + index / 8 * plotWidth;
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', x);
    line.setAttribute('x2', x);
    line.setAttribute('y1', 0);
    line.setAttribute('y2', height);
    line.setAttribute('class', 'roll-time-grid');
    svg.append(line);
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
    const choose = () => auditionNote(note);
    rect.addEventListener('click', choose);
    rect.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(); }
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
        event.preventDefault();
        navigateScore(note, event.key);
      }
    });
    svg.append(rect);

    if (note.waveform?.length > 1) {
      const noteHeight = Math.max(4, rowHeight - 2);
      const center = y + noteHeight / 2;
      const amplitude = noteHeight * 0.36;
      const top = note.waveform.map((value, index) => [
        x + index / (note.waveform.length - 1) * noteWidth,
        center - value * amplitude,
      ]);
      const bottom = [...note.waveform].reverse().map((value, reverseIndex) => {
        const index = note.waveform.length - 1 - reverseIndex;
        return [x + index / (note.waveform.length - 1) * noteWidth, center + value * amplitude];
      });
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', [...top, ...bottom].map(([px, py], index) => `${index ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join(' ') + ' Z');
      path.setAttribute('class', 'roll-waveform');
      path.dataset.event = note.event;
      svg.append(path);
    }
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

ui.roll.addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
  if (event.target?.classList?.contains('roll-note')) return;
  const selected = state.example.notes.find((note) => note.event === state.selectedEvent);
  if (!selected) return;
  event.preventDefault();
  navigateScore(selected, event.key);
});

function selectChannel(id, onReady = null) {
  const channel = currentChannel(id);
  if (!channel) return;
  const wasPlaying = !audio.paused;
  const position = audio.currentTime || 0;
  auditionAudio.pause();
  state.auditionOffset = null;
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
  ui.image.src = mixtureChannel().spectrogram;
  ui.channelLabel.textContent = `Mixture spectrogram · listening: ${channel.label}`;
  document.querySelectorAll('.source-button').forEach((button) => {
    button.setAttribute('aria-pressed', String(button.dataset.channel === id));
  });
}

function selectExample(id) {
  audio.pause();
  state.segmentEnd = null;
  state.example = state.manifest.examples.find((example) => example.id === id);
  state.channel = 'aso';
  state.selectedEvent = state.example.targetEvent;
  ui.title.textContent = state.example.title;
  ui.instrument.textContent = state.example.instrument;
  ui.request.textContent = state.example.qualitative
    ? `${state.example.requestLabel} · queried pitch ${pitchName(state.example.targetPitch)}`
    : `Frozen request ${state.example.requestId} · queried pitch ${pitchName(state.example.targetPitch)}`;
  ui.duration.textContent = state.example.duration.toFixed(2);
  audio.src = currentChannel().audio;
  audio.load();
  ui.image.src = mixtureChannel().spectrogram;
  ui.channelLabel.textContent = 'Mixture spectrogram · listening: ASO';
  ui.binCandidates.hidden = true;
  loadHitMap();
  renderTabs();
  renderChannels();
  renderScore();
  const target = state.example.notes.find((note) => note.target);
  setSelection(target.start, target.end, target.event);
}

function play(start, end = null) {
  auditionAudio.pause();
  state.auditionOffset = null;
  audio.currentTime = clamp(start, 0, state.example.duration);
  state.segmentEnd = end;
  audio.play().catch((error) => console.warn('Playback was blocked:', error));
}

function stop() {
  audio.pause();
  auditionAudio.pause();
  state.auditionOffset = null;
  state.segmentEnd = null;
}

function updatePlayhead() {
  if (!state.example) return;
  if (state.segmentEnd !== null && audio.currentTime >= state.segmentEnd) stop();
  const displayTime = state.auditionOffset === null
    ? (audio.currentTime || 0)
    : state.auditionOffset + (auditionAudio.currentTime || 0);
  const ratio = clamp(displayTime / state.example.duration, 0, 1);
  ui.playhead.style.left = `${ratio * 100}%`;
  ui.currentTime.textContent = displayTime.toFixed(2);
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

ui.playWhole.addEventListener('click', () => play(0));
ui.playRegion.addEventListener('click', () => play(state.start, state.end));
ui.stop.addEventListener('click', stop);

ui.maskToggle.addEventListener('change', () => {
  const hideMask = !ui.maskToggle.checked || !ui.mask.src;
  ui.mask.classList.toggle('mask-overlay--hidden', hideMask);
  ui.maskDimmer.classList.toggle('mask-dimmer--hidden', hideMask);
});

ui.spectrogram.addEventListener('pointerdown', (event) => {
  state.pointerStart = {
    x: event.clientX,
    y: event.clientY,
    time: spectrogramTime(event),
  };
  state.dragging = false;
  ui.spectrogram.setPointerCapture(event.pointerId);
});
ui.spectrogram.addEventListener('pointermove', (event) => {
  if (!state.pointerStart) return;
  if (Math.abs(event.clientX - state.pointerStart.x) > 6) state.dragging = true;
  if (state.dragging) setSelection(state.pointerStart.time, spectrogramTime(event));
});
ui.spectrogram.addEventListener('pointerup', (event) => {
  if (!state.pointerStart) return;
  if (state.dragging) setSelection(state.pointerStart.time, spectrogramTime(event));
  else selectSpectrogramPoint(event);
  state.pointerStart = null;
  state.dragging = false;
});
ui.spectrogram.addEventListener('pointercancel', () => {
  state.pointerStart = null;
  state.dragging = false;
});

auditionAudio.addEventListener('ended', () => {
  state.auditionOffset = null;
});

fetch('assets/manifest.json?v=20260912-0345')
  .then((response) => {
    if (!response.ok) throw new Error(`Could not load demo manifest (${response.status})`);
    return response.json();
  })
  .then((manifest) => {
    state.manifest = manifest;
    selectExample(manifest.examples[0].id);
    updatePlayhead();
  })
  .catch((error) => {
    ui.title.textContent = 'The demo assets could not be loaded.';
    ui.request.textContent = 'Serve this folder through a web server or GitHub Pages.';
    console.error(error);
  });
