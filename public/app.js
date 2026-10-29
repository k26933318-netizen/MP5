const app = document.querySelector('#app');
let election;
let candidates = [];
let voter = '';
let sessionToken = '';
let currentUser;
let activeCameraStream;
let faceAssessmentTimer;
let faceChallengeTimer;
let verificationToken = '';
let ledgerMode = 'fabric';
let faceVerificationEnabled = true;
let resultsLoadSequence = 0;
let resultsRefreshTimer;

function stopCamera() {
  clearInterval(faceAssessmentTimer);
  clearInterval(faceChallengeTimer);
  faceAssessmentTimer = null;
  faceChallengeTimer = null;
  if (activeCameraStream) activeCameraStream.getTracks().forEach(track => track.stop());
  activeCameraStream = null;
}
const view = html => {
  stopCamera();
  clearInterval(resultsRefreshTimer);
  resultsRefreshTimer = null;
  app.innerHTML = `<div class="view">${html}</div>`;
};
const request = async (path, options = {}) => {
  const isFormData = options.body instanceof FormData;
  const response = await fetch(`/api/blockchain${path}`, {
    ...options,
    headers: {
      ...(options.body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
      ...(sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {}),
      ...(options.headers || {})
    }
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || 'Request failed.');
  return data;
};

async function loadLedgerMode() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    ledgerMode = data.blockchain || 'fabric';
  } catch (error) {
    console.warn('Could not load ledger mode:', error);
  }
}

function portalInitializationError(error) {
  view('<div class="kicker">Sign-in successful</div><h2>Voting portal could not load.</h2><p class="intro">Your account is signed in, but a required service could not load the portal.</p><div class="notice error" id="portal-error"></div><button class="button full" id="retry-portal">Retry</button><button class="button secondary full" id="portal-logout">Sign out</button>');
  document.querySelector('#portal-error').textContent = error.message;
  document.querySelector('#retry-portal').onclick = () => start().catch(portalInitializationError);
  document.querySelector('#portal-logout').onclick = logout;
}

async function start() {
  if (!sessionToken) return loginPage();
  if (currentUser.role === 'admin') return adminDashboard();
  if (currentUser.role === 'election_officer') return electionOfficerDashboard();
  if (currentUser.role === 'candidate') return candidateDashboard();
  const data = await request('/elections');
  const now = Date.now();
  election = data.elections.find(item => item.status === 'ACTIVE' &&
    now >= Date.parse(item.startDate) && now <= Date.parse(item.endDate));
  if (!election) {
    view(`<div class="kicker">Voter portal</div><h2>No active election</h2><p class="intro">There is no election currently open for voting.</p><button class="button secondary full" onclick="logout()">Sign out</button>`);
    return;
  }
  const candidateData = await request(`/election/${election.electionId}/candidates`);
  candidates = candidateData.candidates;
  const [face, faceSetting] = await Promise.all([
    request('/face/status'),
    request('/face-verification/status').catch(() => ({ enabled: true }))
  ]);
  if (!face.faceEnrollment.enrolled && faceSetting.enabled !== false) faceEnrollment();
  else registration();
}

async function electionOfficerDashboard() {
  try {
    const [electionsData, votersData, partiesData, candidatesData] = await Promise.all([
      request('/elections').catch(() => ({ elections: [] })),
      request('/voters').catch(() => ({ voters: [] })),
      request('/parties').catch(() => ({ parties: [] })),
      request('/candidate-accounts').catch(() => ({ candidates: [] }))
    ]);
    const elections = electionsData.elections || [];
    const voters = votersData.voters || [];
    const parties = partiesData.parties || [];
    const candidateAccounts = candidatesData.candidates || [];
    view(`<div class="kicker">Election officer console</div><h2>Election oversight</h2><p class="intro">Review voter registrations, approve eligible voters, and add candidate accounts.</p><div class="receipt"><div class="receipt-line"><span>Signed in as</span><strong>${currentUser.name || currentUser.username}</strong></div><div class="receipt-line"><span>Role</span><strong>Election Officer</strong></div></div><h3>Election summary</h3><div class="receipt"><div class="receipt-line"><span>Total elections</span><strong>${elections.length}</strong></div><div class="receipt-line"><span>Approved voters</span><strong>${voters.filter(v => v.status === 'APPROVED').length}</strong></div><div class="receipt-line"><span>Pending approvals</span><strong>${voters.filter(v => v.status !== 'APPROVED').length}</strong></div></div><h3>Voter approvals</h3><div class="admin-list">${voters.length ? voters.map(item => `<div class="admin-row"><span><strong>${item.name}</strong><small>${item.voterId} · ${item.status}</small></span>${item.status === 'APPROVED' ? '<span class="approved">Approved</span>' : `<button class="button small" data-officer-approve="${item.voterId}">Approve</button>`}</div>`).join('') : '<p class="intro">No voter registrations yet.</p>'}</div><h3>Add candidate account</h3><form id="officer-candidate-form" class="form-grid"><div class="field"><label>Candidate ID</label><input id="officer-candidate-id" required placeholder="CAND-001"></div><div class="field"><label>Candidate name</label><input id="officer-candidate-name" required placeholder="Candidate full name"></div><div class="field"><label>Username</label><input id="officer-candidate-username" required placeholder="Candidate username"></div><div class="field"><label>Email</label><input id="officer-candidate-email" type="email" placeholder="candidate@example.com"></div><div class="field"><label>Phone</label><input id="officer-candidate-phone" placeholder="Phone number"></div><div class="field"><label>Party</label><select id="officer-candidate-party" required>${parties.length ? parties.map(party => `<option value="${party.partyId}">${party.partyName}</option>`).join('') : '<option value="">No parties available</option>'}</select></div><div class="field"><label>Election ID</label><input id="officer-candidate-election" required value="${elections[0]?.electionId || ''}" placeholder="Election ID"></div><div class="field"><label>Password</label><input id="officer-candidate-password" type="password" required placeholder="Candidate password"></div><button class="button full" ${parties.length ? '' : 'disabled'}>Add candidate account</button></form><div class="admin-list">${candidateAccounts.length ? candidateAccounts.map(candidate => `<div class="admin-row"><span><strong>${candidate.name || candidate.candidateName}</strong><small>${candidate.username} · ${candidate.partyName || 'Unassigned party'}</small></span><span class="approved">${candidate.status || 'APPROVED'}</span></div>`).join('') : '<p class="intro">No candidate accounts created yet.</p>'}</div><h3>Election records</h3><div class="admin-list">${elections.length ? elections.map(item => `<div class="admin-row"><span><strong>${item.electionName}</strong><small>${item.electionId} · ${item.status}</small></span><span class="approved">${new Date(item.startDate).toLocaleDateString()}</span></div>`).join('') : '<p class="intro">No elections created.</p>'}</div><button class="button secondary full" id="officer-logout">Sign out</button>`);
    document.querySelector('#officer-logout').onclick = logout;
    document.querySelectorAll('[data-officer-approve]').forEach(button => button.onclick = async () => {
      button.disabled = true;
      try {
        await request(`/voters/${encodeURIComponent(button.dataset.officerApprove)}/approve`, { method: 'PATCH', body: JSON.stringify({ status: 'APPROVED' }) });
        electionOfficerDashboard();
      } catch (error) {
        button.disabled = false;
        button.insertAdjacentHTML('afterend', `<small class="logo-upload-error">${error.message}</small>`);
      }
    });
    document.querySelector('#officer-candidate-form').onsubmit = async event => {
      event.preventDefault();
      const form = event.currentTarget;
      try {
        await request('/candidate-accounts', { method: 'POST', body: JSON.stringify({ candidateId: document.querySelector('#officer-candidate-id').value.trim(), name: document.querySelector('#officer-candidate-name').value.trim(), username: document.querySelector('#officer-candidate-username').value.trim(), email: document.querySelector('#officer-candidate-email').value.trim(), phone: document.querySelector('#officer-candidate-phone').value.trim(), partyId: document.querySelector('#officer-candidate-party').value, electionId: document.querySelector('#officer-candidate-election').value.trim(), password: document.querySelector('#officer-candidate-password').value }) });
        electionOfficerDashboard();
      } catch (error) { form.insertAdjacentHTML('afterend', `<div class="notice error">${error.message}</div>`); }
    };
  } catch (error) {
    view(`<div class="kicker">Officer dashboard unavailable</div><h2>${error.message}</h2><button class="button full" onclick="logout()">Back to login</button>`);
  }
}
async function candidateDashboard() {
  try {
    const [partiesData, electionsData] = await Promise.all([
      request('/parties').catch(() => ({ parties: [] })),
      request('/elections').catch(() => ({ elections: [] }))
    ]);
    const parties = partiesData.parties || [];
    const elections = electionsData.elections || [];
    const party = parties.find(item => item.partyId === currentUser.partyId) || null;
    view(`<div class="kicker">Candidate portal</div><h2>Candidate profile</h2><p class="intro">Your party is managed by the administrator. You can review your campaign profile and election schedule here.</p><div class="receipt"><div class="receipt-line"><span>Candidate</span><strong>${currentUser.name || currentUser.username}</strong></div><div class="receipt-line"><span>Username</span><strong>${currentUser.username}</strong></div><div class="receipt-line"><span>Assigned party</span><strong>${party ? party.partyName : 'Awaiting assignment'}</strong></div></div><h3>Available parties</h3><div class="admin-list">${parties.length ? parties.map(item => `<div class="admin-row"><span><strong>${item.partyName}</strong><small>${item.partyId} · ${item.partySymbol || 'Party'}</small></span><span class="approved">${item.presidentName || 'Leader'}</span></div>`).join('') : '<p class="intro">No parties available yet.</p>'}</div><h3>Election schedule</h3><div class="admin-list">${elections.length ? elections.map(item => `<div class="admin-row"><span><strong>${item.electionName}</strong><small>${item.electionId}</small></span><span class="approved">${item.status}</span></div>`).join('') : '<p class="intro">No elections published yet.</p>'}</div><button class="button secondary full" id="candidate-logout">Sign out</button>`);
    document.querySelector('#candidate-logout').onclick = logout;
  } catch (error) {
    view(`<div class="kicker">Candidate portal unavailable</div><h2>${error.message}</h2><button class="button full" onclick="logout()">Back to login</button>`);
  }
}

function loginPage() {
  view(`<div class="kicker">Secure access</div><h2>Sign in to VoteLedger</h2><p class="intro">Choose your account type. Administrators manage parties, alliances, officers, candidates, and elections; voters receive a single ballot.</p><div class="auth-tabs"><button class="auth-tab active" data-role="voter">Voter login</button><button class="auth-tab" data-role="admin">Admin login</button><button class="auth-tab" data-role="election_officer">Election officer login</button><button class="auth-tab" data-role="candidate">Candidate login</button></div><form id="login-form" class="form-grid"><div class="field"><label for="username" id="username-label">Voter ID</label><input id="username" required placeholder="Enter your voter ID"></div><div class="field"><label for="password">Password</label><input id="password" type="password" required placeholder="Enter your password"></div><button class="button full">Continue securely</button></form><div class="meta-row"><span>New voter?</span><span class="inline-link" id="register">Register here</span></div><div id="login-error"></div>`);
  let role = 'voter';
  document.querySelectorAll('.auth-tab').forEach(tab => tab.onclick = () => {
    role = tab.dataset.role;
    document.querySelectorAll('.auth-tab').forEach(item => {
      item.classList.toggle('active', item === tab);
      item.setAttribute('aria-pressed', String(item === tab));
    });
    const labelMap = {
      admin: 'Admin username',
      election_officer: 'Election officer username',
      candidate: 'Candidate username',
      voter: 'Voter ID'
    };
    const placeholderMap = {
      admin: 'Enter admin username',
      election_officer: 'Enter officer username',
      candidate: 'Enter candidate username',
      voter: 'Enter your voter ID'
    };
    document.querySelector('#username-label').textContent = labelMap[role];
    document.querySelector('#username').placeholder = placeholderMap[role];
    document.querySelector('#register').parentElement.style.display = role === 'voter' ? 'flex' : 'none';
  });
  document.querySelector('#register').onclick = voterRegistration;
  document.querySelector('#login-form').onsubmit = async event => {
    event.preventDefault();
    let data;
    try {
      data = await request('/auth/login', { method: 'POST', body: JSON.stringify({ role, username: document.querySelector('#username').value.trim(), password: document.querySelector('#password').value }) });
    } catch (error) {
      const errorContainer = document.querySelector('#login-error');
      if (errorContainer) errorContainer.innerHTML = `<div class="notice error">${error.message}</div>`;
      return;
    }
    sessionToken = data.token;
    currentUser = data.user;
    try {
      await start();
    } catch (error) {
      portalInitializationError(error);
    }
  };
}

function voterRegistration() {
  view(`<div class="kicker">Voter registration</div><h2>Create your voter account</h2><p class="intro">Complete your profile, then capture a short face sequence as a required registration step. Your account is created only after the server-side checks succeed and must be approved before voting.</p><form id="register-form" class="form-grid"><div class="field"><label>Voter ID</label><input id="new-voter-id" required placeholder="e.g. VOTER001"></div><div class="field"><label>Full name</label><input id="new-name" required placeholder="Your full name"></div><div class="field"><label>Date of birth</label><input id="new-dob" type="date" required></div><div class="field"><label>Gender</label><select id="new-gender" required><option value="">Select gender</option><option>Male</option><option>Female</option><option>Other</option><option>Prefer not to say</option></select></div><div class="field"><label>Email address</label><input id="new-email" type="email" placeholder="name@example.com"></div><div class="field"><label>Mobile number</label><input id="new-phone" required placeholder="+91 00000 00000"></div><div class="field"><label>Aadhaar number</label><input id="new-aadhaar" required placeholder="XXXX XXXX XXXX"></div><div class="field"><label>Street</label><input id="new-street" required placeholder="House number, street name"></div><div class="field"><label>City</label><input id="new-city" required placeholder="Your city"></div><div class="field"><label>District</label><input id="new-district" required placeholder="Your district"></div><div class="field"><label>Pincode</label><input id="new-pincode" required placeholder="Postal code"></div><div class="field"><label>Password</label><input id="new-password" type="password" required minlength="6" placeholder="Create a password"></div><label class="consent"><input id="face-consent" type="checkbox" required><span>I consent to processing live face frames for quality checks, face matching, duplicate-registration prevention, and voting verification. Frames are not retained; an encrypted face template is stored for these purposes.</span></label><button class="button full">Continue to face capture</button></form><div class="meta-row"><span>Already registered?</span><span class="inline-link" id="back-login">Return to login</span></div>`);
  document.querySelector('#back-login').onclick = loginPage;
  document.querySelector('#register-form').onsubmit = async event => { event.preventDefault(); try {
    const form = event.currentTarget;
    const registration = {
      voterId: form.querySelector('#new-voter-id').value.trim(),
      name: form.querySelector('#new-name').value.trim(),
      dob: form.querySelector('#new-dob').value,
      gender: form.querySelector('#new-gender').value,
      email: form.querySelector('#new-email').value.trim(),
      phone: form.querySelector('#new-phone').value.trim(),
      aadhaarRef: form.querySelector('#new-aadhaar').value.trim(),
      address: {
        street: form.querySelector('#new-street').value.trim(),
        city: form.querySelector('#new-city').value.trim(),
        district: form.querySelector('#new-district').value.trim(),
        pincode: form.querySelector('#new-pincode').value.trim()
      },
      password: form.querySelector('#new-password').value,
      biometricConsent: form.querySelector('#face-consent').checked
    };
    cameraWorkflow({
      title: 'Capture your face to register',
      intro: 'Center your face in the guide and hold still while the server checks image quality and presentation-attack risk. Only an encrypted face template is saved.',
      endpoint: '/voters',
      fields: { registration: JSON.stringify(registration) },
      actionLabel: 'Check face and register',
      mode: 'registration',
      onSuccess: () => view(`<div class="success-mark">✓</div><div class="kicker">Registration successful</div><h2>Your voter account is registered</h2><p class="intro">Your account and encrypted face template have been saved. An administrator must approve the account before you can log in and vote.</p><button class="button full" onclick="loginPage()">Return to login</button>`),
      onBack: voterRegistration
    });
  } catch (error) { view(`<div class="kicker">Registration failed</div><h2>Could not create account</h2><div class="notice error">${error.message}</div><button class="button" onclick="voterRegistration()">Try again</button>`); } };
}

function safePartyImage(value) {
  return typeof value === 'string' &&
    /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(value)
    ? value
    : '';
}

function partyLogoElement(name, imageData, className = 'party-logo', symbol = '') {
  const source = safePartyImage(imageData);
  if (source) {
    const image = document.createElement('img');
    image.className = className;
    image.src = source;
    image.alt = `${name} logo`;
    return image;
  }
  const emblem = {
    'rising sun': '<circle cx="24" cy="24" r="9"/><path d="M24 4v5M24 39v5M4 24h5M39 24h5M9.9 9.9l3.5 3.5m21.2 21.2 3.5 3.5m0-28.2-3.5 3.5M13.4 34.6l-3.5 3.5"/>',
    lamp: '<path d="M16 20a8 8 0 1 1 16 0c0 3-2 4-3 7H19c-1-3-3-4-3-7Zm4 11h8m-7 4h6m-3-31v4m-14 5 4 2m24-2-4 2"/>',
    tree: '<path d="m24 5 12 15h-7l9 11H26v9h-4v-9H10l9-11h-7L24 5Z"/>'
  }[String(symbol).trim().toLowerCase()];
  if (emblem) {
    const mark = document.createElement('span');
    mark.className = `${className} party-logo-placeholder party-logo-emblem`;
    mark.innerHTML = `<svg viewBox="0 0 48 48" aria-hidden="true">${emblem}</svg>`;
    mark.setAttribute('aria-label', `${name} ${symbol} party symbol`);
    return mark;
  }
  const initials = String(name || 'Party')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map(word => word[0].toUpperCase())
    .join('');
  const placeholder = document.createElement('span');
  placeholder.className = `${className} party-logo-placeholder`;
  placeholder.textContent = initials || 'P';
  placeholder.setAttribute('aria-label', `${name || 'Party'} logo not provided`);
  return placeholder;
}

function setupPartyLogoControls(parties) {
  const form = document.querySelector('#party-form');
  const submit = form?.querySelector('button');
  if (!form || !submit) return;

  const field = document.createElement('div');
  field.className = 'field logo-upload-field';
  const label = document.createElement('label');
  label.htmlFor = 'party-image';
  label.textContent = 'Party logo';
  const input = document.createElement('input');
  input.id = 'party-image';
  input.type = 'file';
  input.accept = 'image/jpeg,image/png,image/webp';
  const preview = document.createElement('div');
  preview.className = 'logo-preview';
  preview.setAttribute('aria-live', 'polite');
  field.append(label, input, preview);
  form.insertBefore(field, submit);
  input.addEventListener('change', () => {
    preview.replaceChildren();
    const file = input.files?.[0];
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) {
      input.value = '';
      preview.textContent = 'Choose a logo smaller than 2 MB.';
      return;
    }
    const image = document.createElement('img');
    image.className = 'party-logo-preview';
    image.alt = 'Selected party logo preview';
    const reader = new FileReader();
    reader.addEventListener('load', () => {
      if (typeof reader.result === 'string') image.src = reader.result;
      preview.replaceChildren(image);
    });
    reader.addEventListener('error', () => {
      preview.textContent = 'Could not preview this image.';
    });
    reader.readAsDataURL(file);
  });

  const heading = [...document.querySelectorAll('.view h3')].find(item => item.textContent === 'Party management');
  const list = heading?.nextElementSibling?.nextElementSibling;
  if (!list) return;
  parties.forEach((party, index) => {
    const row = list.children[index];
    if (row?.classList.contains('admin-row')) {
      row.prepend(partyLogoElement(party.partyName, party.imageData));
      const label = document.createElement('label');
      label.className = 'button small logo-upload-action';
      label.textContent = party.imageData ? 'Replace logo' : 'Add logo';
      const logoInput = document.createElement('input');
      logoInput.type = 'file';
      logoInput.accept = 'image/jpeg,image/png,image/webp';
      logoInput.setAttribute('aria-label', `Upload ${party.partyName} logo`);
      label.append(logoInput);
      const errorMessage = document.createElement('small');
      errorMessage.className = 'logo-upload-error';
      logoInput.addEventListener('change', async () => {
        errorMessage.textContent = '';
        const file = logoInput.files?.[0];
        if (!file) return;
        if (file.size > 2 * 1024 * 1024) {
          logoInput.value = '';
          errorMessage.textContent = 'Choose a logo smaller than 2 MB.';
          return;
        }
        const form = new FormData();
        form.append('image', file);
        try {
          await request(`/parties/${encodeURIComponent(party.partyId)}/logo`, { method: 'PATCH', body: form });
          await adminDashboard();
        } catch (error) {
          errorMessage.textContent = error.message;
        }
      });
      row.append(label, errorMessage);
    }
  });
}

function createAccountDialog(title, description) {
  const dialog = document.createElement('dialog');
  dialog.className = 'account-dialog';
  dialog.setAttribute('aria-labelledby', 'account-dialog-title');
  const panel = document.createElement('div');
  panel.className = 'account-dialog-panel';
  const heading = document.createElement('div');
  heading.className = 'account-dialog-heading';
  const copy = document.createElement('div');
  const titleElement = document.createElement('h2');
  titleElement.id = 'account-dialog-title';
  titleElement.textContent = title;
  const descriptionElement = document.createElement('p');
  descriptionElement.textContent = description;
  copy.append(titleElement, descriptionElement);
  const close = document.createElement('button');
  close.className = 'button secondary account-dialog-close';
  close.type = 'button';
  close.setAttribute('aria-label', 'Close dialog');
  close.textContent = 'Close';
  close.addEventListener('click', () => dialog.close());
  heading.append(copy, close);
  panel.append(heading);
  dialog.append(panel);
  app.querySelector('.view').append(dialog);
  dialog.addEventListener('close', () => dialog.remove(), { once: true });
  dialog.addEventListener('click', event => {
    if (event.target === dialog) dialog.close();
  });
  dialog.showModal();
  return { dialog, panel };
}

function appendAccountInput(form, { name, label, value = '', type = 'text', required = false, options = null }) {
  const field = document.createElement('div');
  field.className = 'field';
  const labelElement = document.createElement('label');
  labelElement.htmlFor = `account-edit-${name}`;
  labelElement.textContent = label;
  let control;
  if (options) {
    control = document.createElement('select');
    options.forEach(optionValue => {
      const option = document.createElement('option');
      option.value = optionValue;
      option.textContent = optionValue || 'Not set';
      control.append(option);
    });
  } else {
    control = document.createElement('input');
    control.type = type;
  }
  control.id = `account-edit-${name}`;
  control.name = name;
  control.value = value == null ? '' : String(value);
  control.required = required;
  field.append(labelElement, control);
  form.append(field);
}

function showAccountDetails(type, item, idField) {
  const labels = {
    voter: [['name', 'Full name'], ['dob', 'Date of birth'], ['gender', 'Gender'], ['phone', 'Phone'], ['email', 'Email'], ['address', 'Address'], ['aadhaarRef', 'Identity reference'], ['status', 'Registration status']],
    officer: [['name', 'Full name'], ['username', 'Username'], ['email', 'Email'], ['contactNumber', 'Contact number'], ['status', 'Account status']],
    candidate: [['name', 'Full name'], ['username', 'Username'], ['email', 'Email'], ['phone', 'Phone'], ['partyName', 'Party'], ['electionId', 'Election'], ['status', 'Account status']]
  };
  const { panel } = createAccountDialog(`${type[0].toUpperCase()}${type.slice(1)} details`, `Account ${item[idField]}`);
  const details = document.createElement('dl');
  details.className = 'account-detail-list';
  labels[type].forEach(([field, label]) => {
    const term = document.createElement('dt');
    term.textContent = label;
    const value = document.createElement('dd');
    let text = item[field];
    if (field === 'address') text = text ? [text.city, text.district, text.state, text.pincode].filter(Boolean).join(', ') : '';
    if (field === 'aadhaarRef' && text) text = `•••• ${String(text).slice(-4)}`;
    value.textContent = text == null || text === '' ? 'Not provided' : String(text);
    details.append(term, value);
  });
  panel.append(details);
}

function editAccount(type, item, idField, route, parties) {
  const fieldsByType = {
    voter: [
      { name: 'name', label: 'Full name', required: true },
      { name: 'dob', label: 'Date of birth', type: 'date' },
      { name: 'gender', label: 'Gender' },
      { name: 'phone', label: 'Phone' },
      { name: 'email', label: 'Email', type: 'email' },
      { name: 'aadhaarRef', label: 'Identity reference' },
      { name: 'city', label: 'City', value: item.address?.city },
      { name: 'district', label: 'District', value: item.address?.district },
      { name: 'state', label: 'State', value: item.address?.state },
      { name: 'pincode', label: 'PIN code', value: item.address?.pincode },
      { name: 'status', label: 'Registration status', options: ['PENDING', 'APPROVED', 'REJECTED'] }
    ],
    officer: [
      { name: 'name', label: 'Full name', required: true },
      { name: 'username', label: 'Username', required: true },
      { name: 'email', label: 'Email', type: 'email' },
      { name: 'contactNumber', label: 'Contact number' },
      { name: 'status', label: 'Account status', options: ['APPROVED', 'SUSPENDED'] }
    ],
    candidate: [
      { name: 'name', label: 'Full name', required: true },
      { name: 'username', label: 'Username', required: true },
      { name: 'email', label: 'Email', type: 'email' },
      { name: 'phone', label: 'Phone' },
      { name: 'partyId', label: 'Party', options: parties.map(party => party.partyId), value: item.partyId },
      { name: 'electionId', label: 'Election ID' },
      { name: 'status', label: 'Account status', options: ['APPROVED', 'SUSPENDED'] }
    ]
  };
  const definitions = fieldsByType[type];
  const { dialog, panel } = createAccountDialog(`Edit ${type} account`, `Update the details for ${item.name || item.candidateName || item.username || item[idField]}.`);
  const form = document.createElement('form');
  form.className = 'account-edit-form';
  definitions.forEach(definition => {
    const currentValue = definition.value !== undefined ? definition.value : item[definition.name];
    const value = definition.type === 'date' && currentValue
      ? String(currentValue).slice(0, 10)
      : definition.options && !definition.options.includes(currentValue)
        ? definition.options[0]
        : currentValue;
    appendAccountInput(form, { ...definition, value });
  });
  const error = document.createElement('div');
  error.className = 'notice error account-edit-error';
  error.setAttribute('role', 'alert');
  error.hidden = true;
  form.append(error);
  const actions = document.createElement('div');
  actions.className = 'account-dialog-actions';
  const cancel = document.createElement('button');
  cancel.className = 'button secondary';
  cancel.type = 'button';
  cancel.textContent = 'Cancel';
  cancel.addEventListener('click', () => dialog.close());
  const save = document.createElement('button');
  save.className = 'button';
  save.type = 'submit';
  save.textContent = 'Save changes';
  actions.append(cancel, save);
  form.append(actions);
  panel.append(form);
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const values = Object.fromEntries(new FormData(form).entries());
    const updates = {};
    definitions.forEach(definition => {
      if (!['city', 'district', 'state', 'pincode'].includes(definition.name)) updates[definition.name] = values[definition.name];
    });
    if (type === 'voter') {
      const address = Object.fromEntries(['city', 'district', 'state', 'pincode'].map(key => [key, values[key].trim()]).filter(([, value]) => value));
      updates.address = Object.keys(address).length ? address : null;
    }
    if (type === 'candidate') updates.candidateName = updates.name;
    save.disabled = true;
    save.textContent = 'Saving…';
    error.hidden = true;
    try {
      await request(`${route}${encodeURIComponent(item[idField])}`, { method: 'PATCH', body: JSON.stringify(updates) });
      dialog.close();
      await adminDashboard();
    } catch (saveError) {
      error.textContent = saveError.message;
      error.hidden = false;
      save.disabled = false;
      save.textContent = 'Save changes';
    }
  });
}

async function adminDashboard() {
  const sidebar = document.querySelector('.sidebar');
  sidebar.classList.add('admin-sidebar');
  sidebar.innerHTML = '<div class="eyebrow">Administration</div><h1>Control<br><em>centre.</em></h1><p class="sidebar-copy">Manage elections, review results, and inspect recorded transactions.</p><nav class="admin-nav" aria-label="Admin sections"><a href="#overview">Overview</a><a href="#results" id="admin-results-link">Results &amp; records</a><a href="#parties">Parties and alliances</a><a href="#access">Officers and candidates</a><a href="#elections">Elections</a><a href="#security">Ledger security</a></nav>';
  const [votersData, electionsData, partiesData, alliancesData, officersData, candidateAccountsData, faceVerificationData] = await Promise.all([
    request('/voters').catch(() => ({ voters: [] })),
    request('/elections').catch(() => ({ elections: [] })),
    request('/parties').catch(() => ({ parties: [] })),
    request('/alliances').catch(() => ({ alliances: [] })),
    request('/officers').catch(() => ({ officers: [] })),
    request('/candidate-accounts').catch(() => ({ candidates: [] })),
    request('/admin/settings/face-verification').catch(() => ({ enabled: true }))
  ]);
  const voters = votersData.voters || [];
  const elections = electionsData.elections || [];
  const parties = partiesData.parties || [];
  const alliances = alliancesData.alliances || [];
  const officers = officersData.officers || [];
  const candidateAccounts = candidateAccountsData.candidates || [];
  const integrity = ledgerMode === 'fabric'
    ? { integrity: { status: 'Fabric peer ledger' } }
    : await request('/ledger/TN-ASSEMBLY-2026/integrity').catch(() => ({ integrity: { status: 'UNKNOWN', recordsChecked: 0, brokenLinks: 0, invalidHashes: 0, verifiedAt: new Date().toISOString() } }));
  view(`<div class="kicker">Admin console</div><h2>Election management</h2><p class="intro">Manage voter approvals, party and alliance records, election officers, and candidate accounts from one control panel.</p><div class="receipt"><div class="receipt-line"><span>Signed in as</span><strong>${currentUser.username}</strong></div><div class="receipt-line"><span>Permission</span><strong>Administrator</strong></div></div><h3>Security overview</h3><div class="receipt"><div class="receipt-line"><span>Total elections</span><strong>${elections.length}</strong></div><div class="receipt-line"><span>Approved voters</span><strong>${voters.filter(v => v.status === 'APPROVED').length}</strong></div><div class="receipt-line"><span>Registered parties</span><strong>${parties.length}</strong></div><div class="receipt-line"><span>Ledger status</span><strong>${integrity.integrity.status}</strong></div></div><h3>Party management</h3><form id="party-form" class="form-grid"><div class="field"><label>Party ID</label><input id="party-id" required placeholder="PARTY-001"></div><div class="field"><label>Party name</label><input id="party-name" required placeholder="Party name"></div><div class="field"><label>Party symbol</label><input id="party-symbol" placeholder="e.g. Lotus"></div><div class="field"><label>President name</label><input id="party-president" placeholder="President name"></div><div class="field"><label>Founded year</label><input id="party-year" type="number" min="1900" max="2100" value="2024"></div><button class="button full">Add party</button></form><div class="admin-list">${parties.length ? parties.map(party => `<div class="admin-row"><span><strong>${party.partyName}</strong><small>${party.partyId}${party.partySymbol ? ` · ${party.partySymbol}` : ''}</small></span><span class="approved">${party.presidentName || 'Party'}</span></div>`).join('') : '<p class="intro">No parties added yet.</p>'}</div><h3>Alliance management</h3><form id="alliance-form" class="form-grid"><div class="field"><label>Alliance ID</label><input id="alliance-id" required placeholder="ALLIANCE-001"></div><div class="field"><label>Alliance name</label><input id="alliance-name" required placeholder="Alliance name"></div><div class="field"><label>Alliance symbol</label><input id="alliance-symbol" placeholder="Symbol"></div><div class="field"><label>Leader party</label><select id="leader-party-id" required>${parties.length ? parties.map(party => `<option value="${party.partyId}">${party.partyName}</option>`).join('') : '<option value="">Add a party first</option>'}</select></div><div class="field"><label>Formed year</label><input id="alliance-year" type="number" min="1900" max="2100" value="2024"></div><button class="button full">Create alliance</button></form><div class="admin-list">${alliances.length ? alliances.map(alliance => `<div class="admin-row"><span><strong>${alliance.allianceName}</strong><small>${alliance.allianceId} · ${alliance.leaderPartyId || 'No leader'}</small></span><span class="approved">${alliance.allianceSymbol || 'Alliance'}</span></div>`).join('') : '<p class="intro">No alliances created yet.</p>'}</div><h3>Election officer management</h3><form id="officer-form" class="form-grid"><div class="field"><label>Officer ID</label><input id="officer-id" required placeholder="OFF-001"></div><div class="field"><label>Name</label><input id="officer-name" required placeholder="Officer name"></div><div class="field"><label>Username</label><input id="officer-username" required placeholder="Officer username"></div><div class="field"><label>Email</label><input id="officer-email" type="email" placeholder="officer@voteledger.gov"></div><div class="field"><label>Contact number</label><input id="officer-phone" placeholder="Contact number"></div><div class="field"><label>Password</label><input id="officer-password" type="password" required placeholder="Secure password"></div><button class="button full">Add officer</button></form><div class="admin-list">${officers.length ? officers.map(officer => `<div class="admin-row"><span><strong>${officer.name}</strong><small>${officer.username} · ${officer.role || 'ELECTION_OFFICER'}</small></span><span class="approved">${officer.status || 'APPROVED'}</span><button class="button small" data-account-view="officer" data-account-id="${officer.officerId}">View</button><button class="button small" data-account-edit="officer" data-account-id="${officer.officerId}">Edit</button></div>`).join('') : '<p class="intro">No officers added yet.</p>'}</div><h3>Candidate account management</h3><form id="candidate-form" class="form-grid"><div class="field"><label>Candidate ID</label><input id="candidate-id" required placeholder="CAND-001"></div><div class="field"><label>Candidate name</label><input id="candidate-name" required placeholder="Candidate full name"></div><div class="field"><label>Username</label><input id="candidate-username" required placeholder="Candidate username"></div><div class="field"><label>Email</label><input id="candidate-email" type="email" placeholder="candidate@example.com"></div><div class="field"><label>Phone</label><input id="candidate-phone" placeholder="Phone number"></div><div class="field"><label>Party</label><select id="candidate-party-id" required>${parties.length ? parties.map(party => `<option value="${party.partyId}">${party.partyName}</option>`).join('') : '<option value="">Add a party first</option>'}</select></div><div class="field"><label>Election ID</label><input id="candidate-election-id" placeholder="TN-ASSEMBLY-2026"></div><div class="field"><label>Password</label><input id="candidate-password" type="password" required placeholder="Candidate password"></div><button class="button full">Add candidate account</button></form><div class="admin-list">${candidateAccounts.length ? candidateAccounts.map(candidate => `<div class="admin-row"><span><strong>${candidate.name || candidate.candidateName}</strong><small>${candidate.username} · ${candidate.partyName || 'Unassigned party'}</small></span><span class="approved">${candidate.status || 'APPROVED'}</span><button class="button small" data-account-view="candidate" data-account-id="${candidate.candidateId}">View</button><button class="button small" data-account-edit="candidate" data-account-id="${candidate.candidateId}">Edit</button></div>`).join('') : '<p class="intro">No candidate accounts created yet.</p>'}</div><h3>Voter approvals</h3><div class="admin-list">${voters.length ? voters.map(voterItem => `<div class="admin-row"><span><strong>${voterItem.name}</strong><small>${voterItem.voterId} · ${voterItem.status}</small></span>${voterItem.status === 'APPROVED' ? '<span class="approved">Approved</span>' : `<button class="button small" data-approve="${voterItem.voterId}">Approve</button>`}<button class="button small" data-account-view="voter" data-account-id="${voterItem.voterId}">View</button><button class="button small" data-account-edit="voter" data-account-id="${voterItem.voterId}">Edit</button></div>`).join('') : '<p class="intro">No voter registrations yet.</p>'}</div><h3>Create election</h3><form id="election-form" class="form-grid"><div class="field"><label>Election ID</label><input id="election-id" required placeholder="ELECTION2026"></div><div class="field"><label>Election name</label><input id="election-name" required placeholder="Election name"></div><div class="field"><label>Start date</label><input id="start-date" type="datetime-local" required></div><div class="field"><label>End date</label><input id="end-date" type="datetime-local" required></div><button class="button full">Create election</button></form><section id="admin-results" class="admin-results"><div class="results-heading"><div><h3>Election results &amp; records</h3><p>Vote totals and anonymized transaction records.</p></div><label for="results-election">Election</label><select id="results-election"></select></div><div id="results-error" class="notice error" role="alert" hidden></div><div id="results-content"><p class="intro">Select an election to load its recorded results.</p></div></section><button class="button secondary full" id="verify-ledger">VERIFY LEDGER</button><button class="button secondary full" id="admin-logout">Sign out</button>`);
  const securityHeading = [...document.querySelectorAll('.view h3')].find(heading => heading.textContent === 'Security overview');
  const faceEnabled = faceVerificationData.enabled !== false;
  if (securityHeading?.nextElementSibling) {
    securityHeading.nextElementSibling.insertAdjacentHTML('afterend', `<section class="receipt face-verification-setting"><div class="receipt-line"><span>Face verification for voting</span><strong id="face-verification-state">${faceEnabled ? 'ON' : 'OFF'}</strong></div><p class="intro">${faceEnabled ? 'Approved voters must pass face and liveness checks before voting.' : 'Approved voters can vote without face verification.'} Disabling this check is recorded in the audit log.</p><button type="button" class="button ${faceEnabled ? 'secondary' : ''}" id="toggle-face-verification">${faceEnabled ? 'Turn face verification off' : 'Turn face verification on'}</button></section>`);
  }
  const faceVerificationButton = document.querySelector('#toggle-face-verification');
  faceVerificationButton?.addEventListener('click', async () => {
    faceVerificationButton.disabled = true;
    try {
      await request('/admin/settings/face-verification', {
        method: 'PATCH',
        body: JSON.stringify({ enabled: !faceEnabled })
      });
      await adminDashboard();
    } catch (error) {
      document.querySelector('#face-verification-state').textContent = `Update failed: ${error.message}`;
      faceVerificationButton.disabled = false;
    }
  });
  setupPartyLogoControls(parties);
  const voterApprovalHeading = [...document.querySelectorAll('.view h3')].find(heading => heading.textContent === 'Voter approvals');
  if (voterApprovalHeading?.nextElementSibling) {
    voterApprovalHeading.nextElementSibling.insertAdjacentHTML('afterend', `<h3>Face enrollment status</h3><div class="admin-list">${voters.length ? voters.map(item => `<div class="admin-row"><span><strong>${item.name}</strong><small>${item.voterId} · ${item.status} · ${item.faceEnrollment?.enrolled ? `Enrolled ${item.faceEnrollment.enrolledAt ? new Date(item.faceEnrollment.enrolledAt).toLocaleDateString() : ''}` : 'Face enrollment required'}</small></span><span class="${item.faceEnrollment?.enrolled ? 'approved' : ''}">${item.faceEnrollment?.enrolled ? 'Enrolled' : 'Required'}</span></div>`).join('') : '<p class="intro">No voters registered yet.</p>'}</div>`);
  }
  document.querySelectorAll('[data-approve]').forEach(button => button.onclick = async () => { await request(`/voters/${button.dataset.approve}/approve`, { method: 'PATCH', body: JSON.stringify({ status: 'APPROVED' }) }); adminDashboard(); });
  const accountCollections = { voter: voters, officer: officers, candidate: candidateAccounts };
  const accountIds = { voter: 'voterId', officer: 'officerId', candidate: 'candidateId' };
  const accountRoutes = { voter: '/voters/', officer: '/officers/', candidate: '/candidate-accounts/' };
  document.querySelectorAll('[data-account-view], [data-account-edit]').forEach(button => {
    const type = button.dataset.accountView || button.dataset.accountEdit;
    const item = accountCollections[type].find(record => record[accountIds[type]] === button.dataset.accountId);
    const name = item?.name || item?.candidateName || item?.username || button.dataset.accountId;
    button.setAttribute('aria-label', `${button.dataset.accountEdit ? 'Edit' : 'View'} ${type} account: ${name}`);
    button.type = 'button';
  });
  document.querySelectorAll('[data-account-view]').forEach(button => button.onclick = () => {
    const type = button.dataset.accountView;
    const item = accountCollections[type].find(record => record[accountIds[type]] === button.dataset.accountId);
    if (item) showAccountDetails(type, item, accountIds[type]);
  });
  document.querySelectorAll('[data-account-edit]').forEach(button => button.onclick = () => {
    const type = button.dataset.accountEdit;
    const item = accountCollections[type].find(record => record[accountIds[type]] === button.dataset.accountId);
    if (item) editAccount(type, item, accountIds[type], accountRoutes[type], parties);
  });
  document.querySelector('#party-form').onsubmit = async event => {
    event.preventDefault();
    try {
      const party = new FormData();
      party.append('partyId', document.querySelector('#party-id').value.trim());
      party.append('partyName', document.querySelector('#party-name').value.trim());
      party.append('partySymbol', document.querySelector('#party-symbol').value.trim());
      party.append('presidentName', document.querySelector('#party-president').value.trim());
      party.append('foundedYear', String(Number(document.querySelector('#party-year').value || new Date().getFullYear())));
      const logo = document.querySelector('#party-image').files?.[0];
      if (logo) party.append('image', logo);
      await request('/parties', { method: 'POST', body: party });
      adminDashboard();
    } catch (error) { document.querySelector('#party-form').insertAdjacentHTML('afterend', `<div class="notice error">${error.message}</div>`); }
  };
  document.querySelector('#alliance-form').onsubmit = async event => {
    event.preventDefault();
    try {
      await request('/alliances', { method: 'POST', body: JSON.stringify({ allianceId: document.querySelector('#alliance-id').value.trim(), allianceName: document.querySelector('#alliance-name').value.trim(), allianceSymbol: document.querySelector('#alliance-symbol').value.trim(), leaderPartyId: document.querySelector('#leader-party-id').value, formedYear: Number(document.querySelector('#alliance-year').value || new Date().getFullYear()) }) });
      adminDashboard();
    } catch (error) { document.querySelector('#alliance-form').insertAdjacentHTML('afterend', `<div class="notice error">${error.message}</div>`); }
  };
  document.querySelector('#officer-form').onsubmit = async event => {
    event.preventDefault();
    try {
      await request('/officers', { method: 'POST', body: JSON.stringify({ officerId: document.querySelector('#officer-id').value.trim(), name: document.querySelector('#officer-name').value.trim(), username: document.querySelector('#officer-username').value.trim(), email: document.querySelector('#officer-email').value.trim(), contactNumber: document.querySelector('#officer-phone').value.trim(), password: document.querySelector('#officer-password').value }) });
      adminDashboard();
    } catch (error) { document.querySelector('#officer-form').insertAdjacentHTML('afterend', `<div class="notice error">${error.message}</div>`); }
  };
  document.querySelector('#candidate-form').onsubmit = async event => {
    event.preventDefault();
    try {
      await request('/candidate-accounts', { method: 'POST', body: JSON.stringify({ candidateId: document.querySelector('#candidate-id').value.trim(), name: document.querySelector('#candidate-name').value.trim(), username: document.querySelector('#candidate-username').value.trim(), email: document.querySelector('#candidate-email').value.trim(), phone: document.querySelector('#candidate-phone').value.trim(), partyId: document.querySelector('#candidate-party-id').value, electionId: document.querySelector('#candidate-election-id').value.trim() || 'TN-ASSEMBLY-2026', password: document.querySelector('#candidate-password').value }) });
      adminDashboard();
    } catch (error) { document.querySelector('#candidate-form').insertAdjacentHTML('afterend', `<div class="notice error">${error.message}</div>`); }
  };
  document.querySelector('#election-form').onsubmit = async event => { event.preventDefault(); try { await request('/elections', { method: 'POST', body: JSON.stringify({ electionId: document.querySelector('#election-id').value.trim(), electionName: document.querySelector('#election-name').value.trim(), startDate: new Date(document.querySelector('#start-date').value).toISOString(), endDate: new Date(document.querySelector('#end-date').value).toISOString(), status: 'ACTIVE' }) }); adminDashboard(); } catch (error) { document.querySelector('#election-form').insertAdjacentHTML('afterend', `<div class="notice error">${error.message}</div>`); } };
  document.querySelector('#admin-results-link').onclick = event => {
    event.preventDefault();
    document.querySelector('#admin-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  };
  const resultsElection = document.querySelector('#results-election');
  elections.forEach(item => {
    const option = document.createElement('option');
    option.value = item.electionId;
    option.textContent = `${item.electionName} (${item.status})`;
    resultsElection.append(option);
  });
  resultsElection.onchange = () => loadAdminResults(resultsElection.value, elections);
  const resultsRefreshButton = document.createElement('button');
  resultsRefreshButton.className = 'button small results-refresh';
  resultsRefreshButton.type = 'button';
  resultsRefreshButton.textContent = 'Refresh now';
  const resultsLiveStatus = document.createElement('span');
  resultsLiveStatus.className = 'results-live-status';
  resultsLiveStatus.setAttribute('aria-live', 'polite');
  resultsLiveStatus.textContent = 'Live updates every 15 seconds';
  document.querySelector('.results-heading').append(resultsLiveStatus, resultsRefreshButton);
  resultsRefreshButton.onclick = () => loadAdminResults(resultsElection.value, elections);
  if (elections.length) {
    loadAdminResults(elections[0].electionId, elections);
    resultsRefreshTimer = setInterval(() => {
      if (resultsElection.isConnected && resultsElection.value) {
        loadAdminResults(resultsElection.value, elections, true);
      }
    }, 15000);
  } else {
    const emptyResults = document.querySelector('#results-content');
    emptyResults.replaceChildren();
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'intro';
    emptyMessage.textContent = 'Create an election to see its live results and recorded votes.';
    emptyResults.append(emptyMessage);
    resultsRefreshButton.disabled = true;
  }
  document.querySelector('#verify-ledger').onclick = async () => {
    if (ledgerMode === 'fabric') {
      view(`<div class="kicker">Fabric ledger</div><h2>Ledger integrity is peer-managed</h2><p class="intro">Hyperledger Fabric validates transactions through its peers and ordering service. This application does not claim to independently verify Fabric block hashes; use your Fabric peer and orderer operational tools for block-level audits.</p><button class="button full" onclick="adminDashboard()">Back to dashboard</button>`);
      return;
    }
    try {
      const result = await request('/ledger/TN-ASSEMBLY-2026/integrity');
      const data = result.integrity;
      view(`<div class="kicker">Ledger verification</div><h2>Integrity report</h2><div class="receipt"><div class="receipt-line"><span>Status</span><strong>${data.status}</strong></div><div class="receipt-line"><span>Records checked</span><strong>${data.recordsChecked}</strong></div><div class="receipt-line"><span>Broken links</span><strong>${data.brokenLinks}</strong></div><div class="receipt-line"><span>Invalid hashes</span><strong>${data.invalidHashes}</strong></div><div class="receipt-line"><span>Verified at</span><strong>${new Date(data.verifiedAt).toLocaleString()}</strong></div></div><button class="button full" onclick="adminDashboard()">Back to dashboard</button>`);
    } catch (error) {
      view(`<div class="kicker">Ledger verification failed</div><h2>${error.message}</h2><button class="button full" onclick="adminDashboard()">Back to dashboard</button>`);
    }
  };
  document.querySelector('#admin-logout').onclick = logout;
}

async function loadAdminResults(electionId, elections, quiet = false) {
  const sequence = ++resultsLoadSequence;
  const errorBox = document.querySelector('#results-error');
  const content = document.querySelector('#results-content');
  if (!errorBox || !content || !electionId) return;
  errorBox.hidden = true;
  errorBox.textContent = '';
  if (!quiet) {
    content.replaceChildren();
    const loading = document.createElement('p');
    loading.className = 'intro';
    loading.textContent = 'Loading election records…';
    content.append(loading);
  }
  try {
    const encodedId = encodeURIComponent(electionId);
    const [resultData, voteData, candidateData] = await Promise.all([
      request(`/results/${encodedId}`),
      request(`/election/${encodedId}/votes`),
      request(`/election/${encodedId}/candidates`)
    ]);
    if (sequence !== resultsLoadSequence || !content.isConnected) return;
    const result = resultData.results;
    const votes = [...voteData.votes].sort((first, second) => Date.parse(second.timestamp) - Date.parse(first.timestamp));
    const candidatesById = new Map(candidateData.candidates.map(candidate => [candidate.candidateId, candidate]));
    const election = elections.find(item => item.electionId === electionId);
    content.replaceChildren();

    const summary = document.createElement('div');
    summary.className = 'explorer-summary';
    [
      ['Election status', election?.status || 'Unavailable'],
      ['Votes recorded', String(result.totalVotesCast)],
      ['Candidates', String(candidateData.candidates.length)]
    ].forEach(([label, value]) => {
      const card = document.createElement('div');
      card.className = 'explorer-stat';
      const caption = document.createElement('span');
      caption.textContent = label;
      const amount = document.createElement('strong');
      amount.textContent = value;
      card.append(caption, amount);
      summary.append(card);
    });
    content.append(summary);

    const totalsHeading = document.createElement('h3');
    totalsHeading.textContent = 'Votes by candidate';
    content.append(totalsHeading);
    const totals = document.createElement('div');
    totals.className = 'candidate-totals';
    const candidates = candidateData.candidates;
    if (!candidates.length) {
      const empty = document.createElement('p');
      empty.className = 'intro';
      empty.textContent = 'No candidates are registered for this election.';
      totals.append(empty);
    } else {
      candidates.forEach(candidate => {
        const count = Number(result.votesByCandidate[candidate.candidateId] || 0);
        const percent = result.totalVotesCast ? (count / result.totalVotesCast) * 100 : 0;
        const row = document.createElement('div');
        row.className = 'candidate-total';
        const identity = document.createElement('div');
        identity.className = 'candidate-total-heading';
        const name = document.createElement('strong');
        name.textContent = candidate.candidateName;
        const detail = document.createElement('span');
        detail.textContent = `${candidate.party || 'Independent'} · ${count} vote${count === 1 ? '' : 's'} · ${percent.toFixed(1)}%`;
        identity.append(name, detail);
        const track = document.createElement('div');
        track.className = 'total-track';
        const bar = document.createElement('span');
        bar.style.width = `${percent}%`;
        track.append(bar);
        row.append(identity, track);
        totals.append(row);
      });
    }
    content.append(totals);

    const recordsHeading = document.createElement('h3');
    recordsHeading.className = 'records-heading';
    recordsHeading.textContent = `Transaction records (${votes.length})`;
    content.append(recordsHeading);
    if (!votes.length) {
      const empty = document.createElement('p');
      empty.className = 'intro';
      empty.textContent = 'No votes have been recorded for this election.';
      content.append(empty);
      return;
    }
    const tableWrap = document.createElement('div');
    tableWrap.className = 'explorer-table-wrap';
    const table = document.createElement('table');
    table.className = 'explorer-table';
    const head = document.createElement('thead');
    head.innerHTML = '<tr><th>Transaction</th><th>Candidate</th><th>Timestamp</th><th>Status</th><th></th></tr>';
    const body = document.createElement('tbody');
    votes.forEach(vote => {
      const candidate = candidatesById.get(vote.candidateId);
      const row = document.createElement('tr');
      const transaction = document.createElement('td');
      transaction.className = 'transaction-cell';
      transaction.textContent = vote.transactionId || 'Unavailable';
      transaction.title = vote.transactionId || '';
      const choice = document.createElement('td');
      choice.textContent = candidate?.candidateName || vote.candidateId || 'Unknown candidate';
      const timestamp = document.createElement('td');
      timestamp.textContent = vote.timestamp ? new Date(vote.timestamp).toLocaleString() : 'Unavailable';
      const status = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = 'transaction-status';
      badge.textContent = vote.status || 'CONFIRMED';
      status.append(badge);
      const action = document.createElement('td');
      const detailsButton = document.createElement('button');
      detailsButton.className = 'button small transaction-details-toggle';
      detailsButton.type = 'button';
      detailsButton.textContent = 'Details';
      action.append(detailsButton);
      row.append(transaction, choice, timestamp, status, action);
      const detailRow = document.createElement('tr');
      detailRow.className = 'transaction-detail-row';
      detailRow.hidden = true;
      const detailCell = document.createElement('td');
      detailCell.colSpan = 5;
      const details = document.createElement('dl');
      details.className = 'transaction-detail-grid';
      [
        ['Transaction ID', vote.transactionId],
        ['Vote receipt', vote.voteId],
        ['Election ID', vote.electionId],
        ['Candidate ID', vote.candidateId],
        ['Record hash', vote.recordHash],
        ['Previous record hash', vote.previousHash]
      ].forEach(([label, value]) => {
        const term = document.createElement('dt');
        term.textContent = label;
        const description = document.createElement('dd');
        description.textContent = value || 'Not provided by this ledger';
        details.append(term, description);
      });
      detailCell.append(details);
      detailRow.append(detailCell);
      detailsButton.onclick = () => {
        detailRow.hidden = !detailRow.hidden;
        detailsButton.textContent = detailRow.hidden ? 'Details' : 'Hide';
      };
      body.append(row, detailRow);
    });
    table.append(head, body);
    tableWrap.append(table);
    content.append(tableWrap);

    const privacy = document.createElement('p');
    privacy.className = 'explorer-privacy';
    privacy.textContent = 'Records are shown without voter IDs, voter hashes, or biometric information.';
    content.append(privacy);
    const liveStatus = document.querySelector('.results-live-status');
    if (liveStatus) liveStatus.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    if (sequence !== resultsLoadSequence || !content.isConnected) return;
    if (!quiet) content.replaceChildren();
    errorBox.textContent = `Could not load election results: ${error.message}`;
    errorBox.hidden = false;
    const liveStatus = document.querySelector('.results-live-status');
    if (liveStatus) liveStatus.textContent = 'Live update failed';
  }
}

async function logout() { await request('/auth/logout', { method: 'POST' }).catch(() => {}); sessionToken = ''; currentUser = null; location.reload(); }

function legacyCameraWorkflow({ title, intro, endpoint, fields = {}, actionLabel, onSuccess, onBack }) {
  view(`<div class="kicker">Face check</div><h2>${title}</h2><p class="intro">${intro}</p><div class="camera"><video id="face-video" autoplay playsinline muted hidden></video><canvas id="face-canvas" hidden></canvas><span id="camera-placeholder">Position your face clearly inside the frame.</span></div><p id="camera-message" class="camera-message" role="status"></p><div class="camera-controls"><button class="button secondary" id="camera-start" type="button">Start camera</button><button class="button" id="camera-capture" type="button" hidden>Capture face</button><button class="button secondary" id="camera-retake" type="button" hidden>Retake</button></div><button class="button full" id="camera-submit" type="button" disabled>${actionLabel}</button><button class="button secondary full" id="camera-back" type="button">Go back</button>`);
  const video = document.querySelector('#face-video');
  const canvas = document.querySelector('#face-canvas');
  const message = document.querySelector('#camera-message');
  const startButton = document.querySelector('#camera-start');
  const captureButton = document.querySelector('#camera-capture');
  const retakeButton = document.querySelector('#camera-retake');
  const submitButton = document.querySelector('#camera-submit');
  let capturedImage = null;

  document.querySelector('#camera-back').onclick = onBack;
  startButton.onclick = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is not available in this browser.');
      activeCameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      const cameraStream = activeCameraStream;
      cameraStream.getVideoTracks().forEach(track => track.addEventListener('ended', () => {
        if (activeCameraStream !== cameraStream) return;
        stopCamera();
        video.hidden = true;
        document.querySelector('#camera-placeholder').hidden = false;
        actionButton.hidden = true;
        startButton.hidden = false;
        startButton.disabled = false;
        showCameraError(new Error('Camera disconnected. Reconnect it and try again.'));
      }, { once: true }));
      video.srcObject = activeCameraStream;
      await video.play();
      video.hidden = false;
      document.querySelector('#camera-placeholder').hidden = true;
      startButton.hidden = true;
      captureButton.hidden = false;
      message.textContent = '';
    } catch (error) {
      stopCamera();
      message.textContent = error.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow camera access and try again.'
        : error.message || 'Camera is unavailable. Check the camera and try again.';
    }
  };
  captureButton.onclick = () => {
    if (!video.videoWidth || !video.videoHeight) {
      message.textContent = 'Camera is not ready yet. Please wait and try again.';
      return;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    stopCamera();
    canvas.toBlob(blob => {
      if (!blob) {
        message.textContent = 'Could not capture the image. Please try again.';
        return;
      }
      capturedImage = blob;
      video.hidden = true;
      canvas.hidden = false;
      captureButton.hidden = true;
      retakeButton.hidden = false;
      submitButton.disabled = false;
      message.textContent = 'Image captured. Review it, then continue.';
    }, 'image/jpeg', 0.92);
  };
  retakeButton.onclick = () => {
    capturedImage = null;
    canvas.hidden = true;
    video.hidden = true;
    document.querySelector('#camera-placeholder').hidden = false;
    retakeButton.hidden = true;
    submitButton.disabled = true;
    startButton.hidden = false;
    message.textContent = '';
  };
  submitButton.onclick = async () => {
    if (!capturedImage) return;
    submitButton.disabled = true;
    message.textContent = 'Checking your face securely...';
    const form = new FormData();
    form.append('image', capturedImage, 'face-capture.jpg');
    for (const [name, value] of Object.entries(fields)) form.append(name, value);
    try {
      const headers = sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {};
      const response = await fetch(`/api/blockchain${endpoint}`, {
        method: 'POST',
        headers,
        body: form
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.message || result.error?.message || 'Face check failed.');
      if (endpoint === '/face/verify' && (!result.verified || !result.verificationToken)) {
        throw new Error('Face verification failed. Please try again with good lighting and keep only one face in view.');
      }
      stopCamera();
      await onSuccess(result);
    } catch (error) {
      message.textContent = error.message;
      submitButton.disabled = false;
    }
  };
}

function cameraWorkflow({ title, intro, endpoint, fields = {}, actionLabel, onSuccess, onBack, mode }) {
  const enrollmentMode = mode === 'enroll' || mode === 'registration';
  const challengeMode = mode === 'verify';
  const cameraTip = enrollmentMode
    ? 'Use even lighting; your background does not need to be white.'
    : 'Use a plain white, evenly lit background.';
  view(`<div class="kicker">Face check</div><h2>${title}</h2><p class="intro">${intro}</p><div class="face-camera-tips" role="note"><strong>Before you start</strong><ul><li>${cameraTip}</li><li>Keep only one face in the camera view.</li><li>Keep your whole face inside the guide and hold still when it turns green.</li></ul></div><div class="camera"><video id="face-video" autoplay playsinline muted hidden></video><canvas id="face-canvas" hidden></canvas><div class="face-guide" id="face-guide" data-state="red" aria-hidden="true"></div><div class="face-tracking-box" id="face-tracking-box" data-state="red" aria-hidden="true" hidden></div><span id="camera-placeholder">Position one face clearly inside the frame.</span></div><p id="camera-message" class="camera-message" data-state="neutral" role="status" aria-live="polite">Start the camera. We’ll guide you through the rest.</p><div id="challenge-prompt" class="challenge-prompt" role="status" aria-live="polite" hidden></div><p id="challenge-countdown" class="challenge-countdown" aria-live="polite"></p><div class="camera-controls"><button class="button secondary" id="camera-start" type="button">Start camera</button><button class="button" id="camera-action" type="button" hidden disabled>${actionLabel}</button><button class="button secondary" id="camera-retry" type="button" hidden>Try face check again</button></div><button class="button secondary full" id="camera-back" type="button">Go back</button>`);
  const video = document.querySelector('#face-video');
  const canvas = document.querySelector('#face-canvas');
  const message = document.querySelector('#camera-message');
  const setCameraMessage = (text, state = 'neutral') => {
    message.textContent = text;
    message.dataset.state = state;
  };
  const cameraErrorMessage = error => /face service did not respond|waking up; retry in a minute|running in render/i.test(error?.message || '')
    ? 'The face check is not responding. Check the service connection and try again.'
    : error?.message || 'The face check failed. Please try again.';
  const guide = document.querySelector('#face-guide');
  const trackingBox = document.querySelector('#face-tracking-box');
  const prompt = document.querySelector('#challenge-prompt');
  const countdown = document.querySelector('#challenge-countdown');
  const startButton = document.querySelector('#camera-start');
  const actionButton = document.querySelector('#camera-action');
  const retryButton = document.querySelector('#camera-retry');
  const assessmentFrames = [];
  const requiredStableFrames = enrollmentMode || challengeMode ? 3 : 2;
  const trackingPaddingFraction = 0.12;
  const stableCenterTolerance = 0.05;
  const stableSizeTolerance = 0.05;
  let readyCount = 0;
  let assessing = false;
  let previousFace = null;
  let pendingTrackingFrame = null;
  let renderTrackingFrame = 0;
  let currentStage = 'position';
  let challenge = null;
  let baselineFrames = [];
  let turnFrames = [];
  let cameraStarting = false;
  let submitting = false;

  document.querySelector('#camera-back').onclick = onBack;

  function renderTrackingBox(face, state) {
    renderTrackingFrame = 0;
    if (!face?.bbox || !video.videoWidth || !video.videoHeight) {
      trackingBox.hidden = true;
      previousFace = null;
      guide.dataset.state = 'red';
      return;
    }
    const cameraRect = video.parentElement.getBoundingClientRect();
    const sourceWidth = Number(face.frameWidth);
    const sourceHeight = Number(face.frameHeight);
    if (!sourceWidth || !sourceHeight || cameraRect.width <= 0 || cameraRect.height <= 0) {
      trackingBox.hidden = true;
      return;
    }
    const scale = Math.max(cameraRect.width / sourceWidth, cameraRect.height / sourceHeight);
    const offsetX = (cameraRect.width - sourceWidth * scale) / 2;
    const offsetY = (cameraRect.height - sourceHeight * scale) / 2;
    const [x1, y1, x2, y2] = face.bbox.map(Number);
    const paddingX = (x2 - x1) * trackingPaddingFraction;
    const paddingY = (y2 - y1) * trackingPaddingFraction;
    const leftInVideo = offsetX + (x1 - paddingX) * scale;
    const rightInVideo = offsetX + (x2 + paddingX) * scale;
    const top = offsetY + (y1 - paddingY) * scale;
    const bottom = offsetY + (y2 + paddingY) * scale;
    trackingBox.style.left = `${cameraRect.width - rightInVideo}px`;
    trackingBox.style.top = `${top}px`;
    trackingBox.style.width = `${rightInVideo - leftInVideo}px`;
    trackingBox.style.height = `${bottom - top}px`;
    trackingBox.dataset.state = state;
    trackingBox.hidden = false;
    guide.dataset.state = state;
  }

  function updateTrackingBox(face, state) {
    pendingTrackingFrame = { face, state };
    if (!renderTrackingFrame) {
      renderTrackingFrame = requestAnimationFrame(() => {
        const pending = pendingTrackingFrame;
        pendingTrackingFrame = null;
        renderTrackingBox(pending?.face, pending?.state || 'red');
      });
    }
  }

  function faceIsStable(face) {
    if (!previousFace) {
      previousFace = face;
      return false;
    }
    const previous = previousFace.bbox;
    const current = face.bbox;
    const width = face.frameWidth;
    const height = face.frameHeight;
    const centerMoved = Math.max(
      Math.abs(((current[0] + current[2]) - (previous[0] + previous[2])) / (2 * width)),
      Math.abs(((current[1] + current[3]) - (previous[1] + previous[3])) / (2 * height))
    );
    const sizeChanged = Math.max(
      Math.abs((current[2] - current[0]) - (previous[2] - previous[0])) / width,
      Math.abs((current[3] - current[1]) - (previous[3] - previous[1])) / height
    );
    previousFace = face;
    return centerMoved <= stableCenterTolerance && sizeChanged <= stableSizeTolerance;
  }

  function loseTracking() {
    previousFace = null;
    updateTrackingBox(null, 'red');
  }

  async function captureBlob() {
    if (!video.videoWidth || !video.videoHeight) throw new Error('The camera is not ready yet. Please wait.');
    const scale = Math.min(1, 640 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext('2d', { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.78));
    if (!blob) throw new Error('Could not capture the image. Please try again.');
    return blob;
  }

  function showCameraError(error) {
    loseTracking();
    guide.dataset.state = 'red';
    setCameraMessage(error.name === 'NotAllowedError'
      ? 'Camera permission was denied. Allow camera access and try again.'
      : error.message || 'Camera is unavailable. Check the camera and try again.', 'red');
  }

  async function assessCurrentFrame() {
    if (assessing || video.hidden || !activeCameraStream || currentStage === 'submitting') return;
    assessing = true;
    const assessmentStream = activeCameraStream;
    const assessmentStage = currentStage;
    let requestTimedOut = false;
    const requestTimer = setTimeout(() => { requestTimedOut = true; }, 12000);
    try {
      const blob = await captureBlob();
      const data = new FormData();
      data.append('image', blob, 'preview.jpg');
      data.append('stage', currentStage);
      data.append('workflow', enrollmentMode ? 'enroll' : 'verify');
      const response = await fetch('/api/blockchain/face/assess', {
        method: 'POST',
        headers: sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {},
        body: data,
        signal: AbortSignal.timeout(12000)
      });
      const result = await response.json();
      if (activeCameraStream !== assessmentStream || video.hidden ||
          currentStage === 'submitting' || currentStage !== assessmentStage) return;
      if (!response.ok) {
        const error = new Error(result.message || result.error?.message || 'Face check is temporarily unavailable.');
        error.status = response.status;
        throw error;
      }
      if (result.face) {
        const stable = result.ready && faceIsStable(result.face);
        if (!result.ready) previousFace = null;
        readyCount = result.ready ? (stable ? readyCount + 1 : 1) : 0;
        updateTrackingBox(result.face, readyCount >= requiredStableFrames ? 'green' : result.state || 'yellow');
      } else {
        loseTracking();
        readyCount = 0;
      }
      if (result.ready && result.face) {
        assessmentFrames.push(blob);
        while (assessmentFrames.length > requiredStableFrames) assessmentFrames.shift();
        if (readyCount >= requiredStableFrames) {
          setCameraMessage(currentStage === 'return'
            ? 'Face centered again. Capture the return-to-center frames.'
            : 'Your face is ready. Hold still.', 'green');
          actionButton.disabled = false;
          actionButton.textContent = currentStage === 'return'
            ? 'Capture return frames'
            : currentStage === 'turn' ? 'Capture head movement'
              : challengeMode ? 'Start liveness challenge' : actionLabel;
        } else {
          setCameraMessage(result.message || 'Hold still while position stabilizes.', 'yellow');
          actionButton.disabled = true;
          if (challengeMode && currentStage === 'position') actionButton.textContent = 'Stabilizing face position';
        }
      } else {
        assessmentFrames.length = 0;
        actionButton.disabled = true;
        setCameraMessage(result.message || 'Center your whole face inside the guide and hold still.', result.state === 'yellow' ? 'yellow' : 'red');
      }
    } catch (error) {
      readyCount = 0;
      assessmentFrames.length = 0;
      actionButton.disabled = true;
      loseTracking();
      if (error.status >= 500 || error.name === 'TypeError' || error.name === 'TimeoutError' || error.name === 'AbortError') {
        setCameraMessage(requestTimedOut || error.name === 'TimeoutError' || error.name === 'AbortError'
          ? 'Face check is taking too long. Check your connection and try again.'
          : 'Face check is temporarily unavailable. Keep the camera open; we’ll retry shortly.', 'yellow');
      } else {
        showCameraError(error);
      }
    } finally {
      clearTimeout(requestTimer);
      assessing = false;
    }
  }

  async function captureBurst(count) {
    const captured = [];
    for (let index = 0; index < count; index += 1) {
      if (index) await new Promise(resolve => setTimeout(resolve, 450));
      captured.push(await captureBlob());
    }
    return captured;
  }

  async function submitFrames(frames, challengeId = '') {
    if (submitting || currentStage === 'submitting') return;
    submitting = true;
    currentStage = 'submitting';
    clearInterval(faceAssessmentTimer);
    faceAssessmentTimer = null;
    clearInterval(faceChallengeTimer);
    faceChallengeTimer = null;
    actionButton.disabled = true;
    while (assessing) await new Promise(resolve => setTimeout(resolve, 50));
    setCameraMessage('Checking image sharpness, face match, and liveness. Hold still.');
    const form = new FormData();
    frames.forEach((frame, index) => form.append('images', frame, `face-${index + 1}.jpg`));
    for (const [name, value] of Object.entries(fields)) form.append(name, value);
    if (challengeId) form.append('challengeId', challengeId);
    try {
      const response = await fetch(`/api/blockchain${endpoint}`, {
        method: 'POST',
        headers: sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {},
        body: form
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.message || result.error?.message || 'Face check failed.');
      if (result.success !== true) throw new Error('The server could not confirm this face check. Please retry.');
      if (['/voters', '/face/enroll'].includes(endpoint) && result.faceEnrollment?.enrolled !== true) {
        throw new Error('The server could not confirm face enrollment. Please retry.');
      }
      if (endpoint === '/face/verify' && (!result.verified || !result.verificationToken)) {
        throw new Error('Face identity or liveness verification failed. Please retry.');
      }
      stopCamera();
      await onSuccess(result);
    } catch (error) {
      currentStage = 'position';
      challenge = null;
      prompt.hidden = true;
      countdown.textContent = '';
      clearInterval(faceChallengeTimer);
      faceChallengeTimer = null;
      retryButton.hidden = false;
      actionButton.hidden = false;
      actionButton.disabled = true;
      setCameraMessage(/sharper image|image is sharp|blurry/i.test(error?.message || '')
        ? 'The image was blurred by movement. Hold still until the guide is green, then try again.'
        : cameraErrorMessage(error), 'red');
      readyCount = 0;
      assessmentFrames.length = 0;
      if (activeCameraStream && !faceAssessmentTimer) {
        faceAssessmentTimer = setInterval(() => assessCurrentFrame(), 1600);
      }
    } finally {
      submitting = false;
    }
  }

  async function startChallenge() {
    actionButton.disabled = true;
    setCameraMessage('Preparing a server-issued liveness challenge...');
    try {
      const response = await fetch('/api/blockchain/face/challenge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {})
        },
        body: JSON.stringify({ electionId: fields.electionId })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.message || 'Could not start the liveness check.');
      challenge = result;
      countdown.textContent = `Challenge expires in ${Math.ceil((Date.parse(result.expiresAt) - Date.now()) / 1000)} seconds.`;
      baselineFrames = assessmentFrames.slice(-result.framesPerStage);
      currentStage = 'turn';
      prompt.hidden = false;
      prompt.textContent = `Step 1 of 2: ${result.action === 'turn_left' ? 'Turn your head toward your left shoulder.' : 'Turn your head toward your right shoulder.'}`;
      setCameraMessage('Keep your face inside the guide while turning.', 'yellow');
      actionButton.textContent = 'Capture head movement';
      actionButton.disabled = true;
      faceChallengeTimer = setInterval(() => {
        const remaining = Math.max(0, Math.ceil((Date.parse(challenge.expiresAt) - Date.now()) / 1000));
        countdown.textContent = `Challenge expires in ${remaining} seconds.`;
        if (!remaining) expireChallenge();
      }, 500);
    } catch (error) {
      showCameraError(error);
      actionButton.disabled = false;
      actionButton.textContent = challengeMode ? 'Start liveness challenge' : actionLabel;
    }
  }

  function expireChallenge() {
    clearInterval(faceChallengeTimer);
    faceChallengeTimer = null;
    challenge = null;
    currentStage = 'position';
    readyCount = 0;
    assessmentFrames.length = 0;
    turnFrames = [];
    baselineFrames = [];
    prompt.hidden = true;
    countdown.textContent = 'Challenge expired. Please start a new check.';
    retryButton.hidden = false;
    actionButton.hidden = true;
    actionButton.disabled = true;
  }

  startButton.onclick = async () => {
    if (cameraStarting || activeCameraStream) return;
    cameraStarting = true;
    startButton.disabled = true;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is not available in this browser.');
      activeCameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      video.srcObject = activeCameraStream;
      await video.play();
      video.hidden = false;
      document.querySelector('#camera-placeholder').hidden = true;
      startButton.hidden = true;
      actionButton.hidden = false;
      actionButton.disabled = true;
      actionButton.textContent = challengeMode ? 'Stabilizing face position' : actionLabel;
      setCameraMessage('Center your face in the guide and hold still.', 'yellow');
      faceAssessmentTimer = setInterval(() => assessCurrentFrame(), enrollmentMode ? 900 : 1600);
      await assessCurrentFrame();
    } catch (error) {
      stopCamera();
      video.srcObject = null;
      video.hidden = true;
      document.querySelector('#camera-placeholder').hidden = false;
      showCameraError(error);
    } finally {
      cameraStarting = false;
      startButton.disabled = Boolean(activeCameraStream);
    }
  };

  actionButton.onclick = async () => {
    if (actionButton.disabled || (assessing && !enrollmentMode) || submitting) return;
    if (!challengeMode) {
      actionButton.disabled = true;
      setCameraMessage('Using the clear frame already checked. Hold still.', 'yellow');
      try {
        const checkedFrames = assessmentFrames.slice(-requiredStableFrames);
        if (checkedFrames.length !== requiredStableFrames || readyCount < requiredStableFrames) {
          throw new Error('Center your whole face in the guide and hold still until it turns green.');
        }
        await submitFrames(checkedFrames);
      } catch (error) {
        actionButton.disabled = false;
        setCameraMessage(error.message || 'Could not capture your face. Please try again.', 'red');
      }
      return;
    }
    if (currentStage === 'position') {
      await startChallenge();
      return;
    }
    if (currentStage === 'turn') {
      actionButton.disabled = true;
      setCameraMessage('Capturing the requested movement. Follow the instruction now.', 'yellow');
      try {
        turnFrames = await captureBurst(challenge.framesPerStage);
        currentStage = 'return';
        readyCount = 0;
        assessmentFrames.length = 0;
        prompt.textContent = 'Step 2 of 2: Return your face to the center and hold still.';
        setCameraMessage('Return to the center. Wait for the guide to turn green.', 'yellow');
      } catch (error) {
        expireChallenge();
        showCameraError(error);
      }
      return;
    }
    if (currentStage === 'return' && challenge) {
      actionButton.disabled = true;
      try {
        const returnFrames = await captureBurst(challenge.framesPerStage);
        const challengeId = challenge.challengeId;
        await submitFrames([...baselineFrames, ...turnFrames, ...returnFrames], challengeId);
      } catch (error) {
        expireChallenge();
        showCameraError(error);
      }
    }
  };

  retryButton.onclick = () => {
    retryButton.hidden = true;
    currentStage = 'position';
    challenge = null;
    prompt.hidden = true;
    countdown.textContent = '';
    readyCount = 0;
    assessmentFrames.length = 0;
    turnFrames = [];
    actionButton.hidden = false;
    actionButton.disabled = true;
    actionButton.textContent = challengeMode ? 'Start liveness challenge' : actionLabel;
    setCameraMessage('Center your face and hold still.', 'yellow');
  };
}

function faceEnrollment() {
  cameraWorkflow({
    title: 'Enroll your face',
    intro: 'Your approved account needs one face enrollment before voting. The image is processed by the private face service; only an encrypted ArcFace embedding is retained.',
    endpoint: '/face/enroll',
    actionLabel: 'Enroll face',
    mode: 'enroll',
    onSuccess: () => start(),
    onBack: logout
  });
}

function registration() {
  view(`<div class="kicker">Voter access</div><h2>Welcome, ${currentUser.name}</h2><p class="intro">Choose a candidate on the ballot. Face verification will follow the current administrator setting.</p><button class="button full" id="continue-verification">Continue to ballot</button><div class="meta-row"><span>Not you?</span><span class="inline-link" id="logout-link">Sign out</span></div>`);
  document.querySelector('#continue-verification').onclick = () => { voter = currentUser.voterId; ballot(); };
  document.querySelector('#logout-link').onclick = logout;
}

async function refreshFaceVerificationSetting() {
  try {
    const setting = await request('/face-verification/status');
    faceVerificationEnabled = setting.enabled !== false;
  } catch {
    faceVerificationEnabled = true;
  }
  return faceVerificationEnabled;
}

async function ballot() {
  await refreshFaceVerificationSetting();
  const faceMessage = faceVerificationEnabled
    ? 'You’ll confirm your choice and complete a face check before your vote is submitted.'
    : 'Face verification is turned off by an administrator for voting right now.';
  view(`<div class="ballot-head"><div><div class="kicker">Your ballot</div><h2>${election.electionName}</h2></div><span class="election-tag">OPEN</span></div><p class="intro">Select one candidate. ${faceMessage}</p><form id="ballot" class="candidates">${candidates.map(candidate => `<label class="candidate"><input type="radio" name="candidate" value="${candidate.candidateId}" required><span><strong>${candidate.candidateName}</strong><small>${candidate.party}</small></span></label>`).join('')}<button class="button full">Review your choice</button></form><div class="notice"><strong>Please choose carefully.</strong><br>You can submit only one vote in this election.</div>`);
  document.querySelectorAll('.candidate').forEach((label, index) => {
    const candidate = candidates[index];
    label.insertBefore(
      partyLogoElement(candidate.party || 'Independent', candidate.partyImage, 'party-logo', candidate.partySymbol),
      label.querySelector('span')
    );
  });
  document.querySelector('#ballot').onsubmit = event => { event.preventDefault(); confirmation(document.querySelector('input[name="candidate"]:checked').value); };
}

async function confirmation(candidateId) {
  await refreshFaceVerificationSetting();
  const candidate = candidates.find(item => item.candidateId === candidateId);
  const confirmationMessage = faceVerificationEnabled
    ? 'Check your selection below. You’ll need to pass a face check before it is submitted.'
    : 'Face verification is disabled by an administrator. Your approved account can submit this vote without a face check.';
  view(`<div class="kicker">Confirm your choice</div><h2>Ready to submit your vote?</h2><p class="intro">${confirmationMessage}</p><div class="receipt"><div class="receipt-line"><span>Election</span><strong>${election.electionName}</strong></div><div class="receipt-line"><span>Your choice</span><strong>${candidate.candidateName}</strong></div></div><button class="button full" id="verify-and-submit">${faceVerificationEnabled ? 'Confirm and continue' : 'Confirm and submit vote'}</button><div class="meta-row"><span>Want to change it?</span><span class="inline-link" id="back">Return to ballot</span></div>`);
  document.querySelector('#back').onclick = ballot;
  const submitVote = async token => {
    const submitButton = document.querySelector('#verify-and-submit');
    submitButton.disabled = true;
    try {
      const data = await request('/vote', {
        method: 'POST',
        body: JSON.stringify({
          electionId: election.electionId,
          candidateId,
          ...(token ? { verificationToken: token } : {})
        })
      });
      verificationToken = '';
      receipt(data.vote || data);
    } catch (error) {
      verificationToken = '';
      view(`<div class="kicker">Vote not submitted</div><h2>We could not record this vote.</h2><div class="notice error">${error.message}</div><button class="button" onclick="ballot()">Return to ballot</button>`);
    }
  };
  document.querySelector('#verify-and-submit').onclick = async () => {
    if (!await refreshFaceVerificationSetting()) {
      await submitVote('');
      return;
    }
    verificationToken = '';
    cameraWorkflow({
      title: 'Verify your face',
      intro: 'Position your face inside the frame. Verification must succeed before the server authorizes this vote.',
      endpoint: '/face/verify',
      fields: { electionId: election.electionId },
      actionLabel: 'Verify and continue',
      mode: 'verify',
      onBack: ballot,
      onSuccess: async result => {
        verificationToken = result.verificationToken;
        await submitVote(verificationToken);
      }
    });
  };
}

function receipt(vote) {
  view(`<div class="success-mark">✓</div><div class="kicker">Vote submitted</div><h2>Your vote is recorded.</h2><p class="intro">Thank you for voting. Your choice has been saved securely.</p><div class="receipt"><div class="receipt-line"><span>Confirmation</span><strong style="color:var(--green)">Successful</strong></div><div class="receipt-line"><span>Receipt number</span><strong>${vote.voteId}</strong></div><div class="receipt-line"><span>Time</span><strong>${new Date(vote.timestamp).toLocaleString()}</strong></div></div><div class="meta-row"><span>Would you like to check it?</span><span class="inline-link" id="verify-receipt-link">Verify receipt</span></div><button class="button secondary full" onclick="location.reload()">Finish</button>`);
  document.querySelector('#verify-receipt-link').onclick = async () => {
    try {
      const result = await request('/receipt/verify', { method: 'POST', body: JSON.stringify({ transactionId: vote.transactionId }) });
      const receiptData = result.receipt;
      view(`<div class="success-mark">✓</div><div class="kicker">Receipt checked</div><h2>Your receipt is valid.</h2><p class="intro">The receipt matches a confirmed vote record.</p><button class="button full" onclick="location.reload()">Finish</button>`);
    } catch (error) {
      view(`<div class="kicker">Receipt verification failed</div><h2>${error.message}</h2><button class="button full" onclick="location.reload()">Close secure session</button>`);
    }
  };
}

loadLedgerMode().then(() => start()).catch(portalInitializationError);
