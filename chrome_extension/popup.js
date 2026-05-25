'use strict';

const API_BASE = 'https://educhat-6lzs.onrender.com';

// ── Region → state mapping ────────────────────────────────────────────────
const REGION_MAP = {
  Northeast: ['MA','CT','RI','NH','VT','ME','NY','NJ','PA','DE','MD'],
  Southeast: ['VA','NC','SC','GA','FL','TN','WV','AL','MS','KY'],
  Midwest:   ['OH','IN','IL','MI','WI','MN','IA','MO','NE','KS','ND','SD'],
  Southwest: ['TX','OK','AR','LA','AZ','NM'],
  West:      ['CA','OR','WA','CO','UT','NV','ID','MT','WY','AK','HI'],
  DC:        ['DC'],
};

const STATE_TO_REGION = {};
for (const [region, states] of Object.entries(REGION_MAP)) {
  states.forEach(s => { STATE_TO_REGION[s] = region; });
}

const SUGGESTION_CHIPS = [
  'What is the application deadline?',
  'What GPA do I need?',
  'What test scores are required?',
  'Is there a separate business school application?',
  'What does the class profile look like?',
];

// ── App state ─────────────────────────────────────────────────────────────
let allColleges = [];
let selectedId  = null;
let isBusy      = false;

// Saved messages for persistence: [{role, text?, html?, sources?}]
let savedMessages = [];

// ── DOM refs ──────────────────────────────────────────────────────────────
const regionSelect    = document.getElementById('regionSelect');
const collegeSelect   = document.getElementById('collegeSelect');
const chatArea        = document.getElementById('chatArea');
const questionInput   = document.getElementById('questionInput');
const sendBtn         = document.getElementById('sendBtn');
const clearBtn        = document.getElementById('clearBtn');
const retryBtn        = document.getElementById('retryBtn');
const loadingSpinner  = document.getElementById('loadingSpinner');

// ── Helpers ───────────────────────────────────────────────────────────────
/**
 * Flip "School Name (University Name)" → "University Name (School Name)".
 * Falls back to the original string if the pattern doesn't match.
 */
function formatCollegeName(displayName) {
  const m = displayName.match(/^(.+?)\s*\((.+?)\)\s*$/);
  return m ? `${m[2]} (${m[1]})` : displayName;
}

function stateOf(location) {
  if (!location) return null;
  const m = location.match(/,\s*([A-Z]{2})$/);
  return m ? m[1] : null;
}

function regionOf(state) {
  return state ? (STATE_TO_REGION[state] || 'Other') : 'Other';
}

function syncInputState() {
  const ready = !!selectedId && !isBusy;
  questionInput.disabled = !ready;
  sendBtn.disabled = !ready || questionInput.value.trim() === '';
  questionInput.placeholder = selectedId
    ? 'Ask an admissions question…'
    : 'Select a college first…';
}

function scrollDown() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── State persistence ─────────────────────────────────────────────────────
function persist() {
  chrome.storage.local.set({
    ec_region:   regionSelect.value,
    ec_college:  selectedId || '',
    ec_messages: savedMessages,
  });
}

async function loadPersistedState() {
  return new Promise(resolve => {
    chrome.storage.local.get(
      ['ec_region', 'ec_college', 'ec_messages'],
      resolve
    );
  });
}

// ── College loading ───────────────────────────────────────────────────────
async function loadColleges(restoreCollegeId) {
  collegeSelect.disabled = true;
  collegeSelect.innerHTML = '<option value="">Loading colleges…</option>';
  retryBtn.classList.add('hidden');
  loadingSpinner.classList.remove('hidden');

  // Show cold-start warning after 6 s
  const coldTimer = setTimeout(() => {
    collegeSelect.innerHTML = '<option value="">Server waking up…</option>';
  }, 6000);

  try {
    const res = await fetch(`${API_BASE}/colleges?indexed_only=true`);
    clearTimeout(coldTimer);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allColleges = await res.json();

    loadingSpinner.classList.add('hidden');
    collegeSelect.disabled = false;
    renderCollegeDropdown(regionSelect.value);

    // Restore previously selected college if any
    if (restoreCollegeId) {
      const found = allColleges.find(c => c.college_id === restoreCollegeId);
      if (found) {
        collegeSelect.value = restoreCollegeId;
        selectedId = restoreCollegeId;
        syncInputState();
      }
    }
  } catch {
    clearTimeout(coldTimer);
    loadingSpinner.classList.add('hidden');
    collegeSelect.innerHTML = '<option value="">Failed — click ↺ to retry</option>';
    retryBtn.classList.remove('hidden');
  }
}

function renderCollegeDropdown(region) {
  const list = region
    ? allColleges.filter(c => regionOf(stateOf(c.location)) === region)
    : allColleges;

  collegeSelect.innerHTML = '<option value="">Select a college…</option>';
  list.forEach((c, i) => {
    const opt = document.createElement('option');
    opt.value = c.college_id;
    const name = formatCollegeName(c.display_name);
    opt.textContent = name;
    collegeSelect.appendChild(opt);
  });

  if (selectedId && list.find(c => c.college_id === selectedId)) {
    collegeSelect.value = selectedId;
  } else if (!list.find(c => c.college_id === selectedId)) {
    selectedId = null;
    syncInputState();
  }
}

// ── Chat rendering ────────────────────────────────────────────────────────
function showCollegeWelcome(college) {
  chatArea.innerHTML = '';
  savedMessages = [];
  persist();

  const wrap = document.createElement('div');
  wrap.className = 'college-welcome';
  wrap.innerHTML = `
    <p>Ask me anything about admissions at<br>
       <strong>${escapeHtml(formatCollegeName(college.display_name))}</strong>.</p>
    <div class="chips" id="chipsRow"></div>
  `;
  chatArea.appendChild(wrap);

  document.getElementById('chipsRow').replaceChildren(
    ...SUGGESTION_CHIPS.map(q => {
      const btn = document.createElement('button');
      btn.className = 'chip';
      btn.textContent = q;
      btn.addEventListener('click', () => {
        questionInput.value = q;
        sendQuestion();
      });
      return btn;
    })
  );
}

function appendUserBubble(text) {
  const el = document.createElement('div');
  el.className = 'bubble bubble-user';
  el.textContent = text;
  chatArea.appendChild(el);
  scrollDown();
}

function appendBotBubble(html, sources = []) {
  // Remove welcome / chips on first real answer
  chatArea.querySelector('.college-welcome')?.remove();

  const el = document.createElement('div');
  el.className = 'bubble bubble-bot';
  el.innerHTML = `<div class="answer-text">${html}</div>`;
  if (sources.length > 0) el.appendChild(buildSourcesBlock(sources));
  chatArea.appendChild(el);
  scrollDown();
}

/** Strip [S#] citations and escape HTML — answer reads as clean prose. */
function formatAnswer(rawText) {
  return escapeHtml(rawText).replace(/\s*\[S\d+\]/g, '');
}

/** Deduplicate sources by domain — one link per website. */
function dedupeSources(sources) {
  const seen = new Set();
  return sources.filter(s => {
    try {
      const hostname = new URL(s.url || s.deep_link).hostname;
      if (seen.has(hostname)) return false;
      seen.add(hostname);
      return true;
    } catch {
      // Malformed URL — fall back to full string dedup
      const key = s.url || s.deep_link;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }
  });
}

function buildSourcesBlock(sources) {
  const unique = dedupeSources(sources);
  if (unique.length === 0) return document.createDocumentFragment();

  const wrap = document.createElement('div');
  wrap.className = 'sources';

  const label = document.createElement('div');
  label.className = 'sources-label';
  label.textContent = 'Sources';
  wrap.appendChild(label);

  const list = document.createElement('ul');
  list.className = 'sources-list open';
  wrap.appendChild(list);

  unique.forEach(s => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.className = 'source-link';
    a.href = s.deep_link;
    a.target = '_blank';
    a.rel = 'noopener';
    // Show the snippet as link text, truncated to ~80 chars
    const text = (s.snippet || s.url || '').trim();
    a.textContent = text.length > 90 ? text.slice(0, 87) + '…' : text;
    a.title = s.url;
    // Prevent popup from opening new tab via <a> — use window.open instead
    a.addEventListener('click', e => {
      e.preventDefault();
      window.open(s.deep_link, '_blank');
    });
    li.appendChild(a);
    list.appendChild(li);
  });

  return wrap;
}

function showTyping() {
  const el = document.createElement('div');
  el.className = 'bubble bubble-bot typing';
  el.id = 'typingBubble';
  el.innerHTML = `
    <div class="typing-dots"><span></span><span></span><span></span></div>
    <span class="typing-label" id="typingLabel"></span>
  `;
  chatArea.appendChild(el);
  scrollDown();

  const timer = setTimeout(() => {
    const lbl = document.getElementById('typingLabel');
    if (lbl) lbl.textContent = 'Waking up server…';
  }, 6000);

  return () => { clearTimeout(timer); el.remove(); };
}

function showError(msg) {
  const el = document.createElement('div');
  el.className = 'error-bubble';
  el.textContent = msg;
  chatArea.appendChild(el);
  scrollDown();
}

// ── Restore saved messages ────────────────────────────────────────────────
function restoreMessages(messages) {
  if (!messages || messages.length === 0) return;
  chatArea.innerHTML = '';
  messages.forEach(m => {
    if (m.role === 'user') {
      appendUserBubble(m.text);
    } else {
      appendBotBubble(m.html || '', m.sources || []);
    }
  });
}

// ── Send question ─────────────────────────────────────────────────────────
async function sendQuestion() {
  const q = questionInput.value.trim();
  if (!q || !selectedId || isBusy) return;

  questionInput.value = '';
  isBusy = true;
  syncInputState();

  // Remove welcome chips if still visible
  chatArea.querySelector('.college-welcome')?.remove();

  appendUserBubble(q);
  savedMessages.push({ role: 'user', text: q });

  const removeTyping = showTyping();

  try {
    const url =
      `${API_BASE}/ask` +
      `?college_id=${encodeURIComponent(selectedId)}` +
      `&question=${encodeURIComponent(q)}`;

    const res = await fetch(url);
    removeTyping();

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      showError(`Error ${res.status}: ${body.detail || res.statusText}`);
    } else {
      const data = await res.json();
      const html = formatAnswer(data.answer);
      // Backend controls exactly which sources to return:
      //   - Cited admission page sources → included when [S#] used
      //   - US News / CDS fallback → included as a single source link
      //   - Irrelevant / not-found → backend returns [] so nothing shows
      const sources = data.sources || [];
      appendBotBubble(html, sources);
      savedMessages.push({ role: 'bot', html, sources });
      persist();
    }
  } catch {
    removeTyping();
    showError('Network error — check your connection and try again.');
  } finally {
    isBusy = false;
    syncInputState();
    questionInput.focus();
  }
}

// ── Event listeners ───────────────────────────────────────────────────────
retryBtn.addEventListener('click', () => loadColleges(selectedId));

regionSelect.addEventListener('change', () => {
  renderCollegeDropdown(regionSelect.value);
  persist();
});

collegeSelect.addEventListener('change', () => {
  selectedId = collegeSelect.value || null;
  syncInputState();
  if (selectedId) {
    const college = allColleges.find(c => c.college_id === selectedId);
    showCollegeWelcome(college);
    questionInput.focus();
  }
  persist();
});

questionInput.addEventListener('input', syncInputState);

questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendQuestion();
  }
});

sendBtn.addEventListener('click', sendQuestion);

clearBtn.addEventListener('click', () => {
  savedMessages = [];
  if (selectedId) {
    const college = allColleges.find(c => c.college_id === selectedId);
    if (college) showCollegeWelcome(college);
  } else {
    chatArea.innerHTML = `
      <div class="welcome-msg">
        <div class="welcome-icon">🎓</div>
        <p class="welcome-title">Welcome to EduChat</p>
        <p class="welcome-desc">Select a college above, then ask any admissions question.</p>
      </div>`;
  }
  persist();
});

// ── Boot ──────────────────────────────────────────────────────────────────
(async () => {
  const saved = await loadPersistedState();

  // Restore region selection first
  if (saved.ec_region) {
    regionSelect.value = saved.ec_region;
  }

  // Load colleges, restoring the saved college selection
  await loadColleges(saved.ec_college);

  // Restore chat messages if we have them and a college is selected
  if (saved.ec_messages && saved.ec_messages.length > 0 && selectedId) {
    savedMessages = saved.ec_messages;
    restoreMessages(savedMessages);
  } else if (selectedId) {
    const college = allColleges.find(c => c.college_id === selectedId);
    if (college) showCollegeWelcome(college);
  }
})();
