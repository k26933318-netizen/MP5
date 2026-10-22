import { FaceCapture } from "/client/face-capture.js";

const mode = document.body.dataset.mode;
if (mode === "enroll" || mode === "identify") {
  initializeCapturePage(mode);
}

function initializeCapturePage(mode) {
  const subjectInput = document.querySelector("#subject");
  const consentInput = document.querySelector("#consent");
  const primaryAction = document.querySelector("#primary-action");
  const actionLabel = document.querySelector("#action-label");
  const spinner = document.querySelector(".button-spinner");
  const cameraFrame = document.querySelector("#camera-frame");
  const cameraStatus = document.querySelector("#camera-status");
  const cameraStatusIcon = document.querySelector("#camera-status-icon");
  const cameraStatusText = document.querySelector("#camera-status-text");
  const result = document.querySelector("#result");
  const resultText = document.querySelector("#result-text");
  let capture = null;
  let sessionId = "";
  let processing = false;

  function setCameraStatus(message, kind = "info", state = "idle") {
    cameraStatus.dataset.kind = kind;
    cameraStatusText.textContent = message;
    cameraStatusIcon.textContent = kind === "success" ? "✓" : kind === "error" ? "!" : "i";
    cameraFrame.dataset.state = state;
  }

  function showResult(message, kind) {
    result.className = `message message--${kind}`;
    resultText.textContent = message;
    result.querySelector(".message-icon").textContent = kind === "success" ? "✓" : kind === "error" ? "!" : "i";
    result.hidden = false;
  }

  function setProcessing(value, label = "Working…") {
    processing = value;
    primaryAction.disabled = value;
    primaryAction.setAttribute("aria-busy", String(value));
    spinner.hidden = !value;
    if (value) actionLabel.textContent = label;
    else refreshActionAvailability();
  }

  function refreshActionAvailability() {
    if (capture && primaryAction.dataset.phase === "capture") {
      primaryAction.disabled = processing || !capture.faceReady;
    }
  }

  async function requestJson(url, options = {}) {
    let response;
    try {
      response = await fetch(url, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) }
      });
    } catch (error) {
      throw new Error("The application service could not be reached. Check that it is running, then retry.");
    }
    let body;
    try {
      body = await response.json();
    } catch (error) {
      throw new Error(`The application returned an unreadable response (${response.status}).`);
    }
    if (!response.ok) throw new Error(body.message || body.public_message || `Request failed (${response.status}).`);
    return body;
  }

  async function startCamera() {
    const subjectId = mode === "enroll" ? subjectInput.value.trim() : "";
    if (mode === "enroll" && !/^[A-Za-z0-9_.:-]{1,128}$/.test(subjectId)) {
      subjectInput.focus();
      throw new Error("Enter an ID using letters, numbers, periods, underscores, colons, or hyphens.");
    }
    if (mode === "enroll" && !consentInput.checked) {
      consentInput.focus();
      throw new Error("Please confirm consent before enrolling your face.");
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera access requires localhost or HTTPS. Open this page from localhost or a secure address.");
    }

    const sessionRequest = { purpose: mode };
    if (mode === "enroll") sessionRequest.subject_id = subjectId;
    const session = await requestJson("/service-api/sessions", {
      method: "POST",
      body: JSON.stringify(sessionRequest)
    });
    sessionId = session.session_id;
    capture = new FaceCapture({
      baseUrl: session.service_url,
      sessionId,
      video: document.querySelector("#video"),
      mode,
      onGuidance: (message, state = {}) => {
        const kind = state.faceReady ? "success" : "info";
        setCameraStatus(message, kind, state.faceReady ? "ready" : "idle");
        refreshActionAvailability();
      },
      onChallenge: (challenge) => {
        setCameraStatus(`Follow the camera prompts: ${challenge.actions.join(" then ").replaceAll("_", " ")}.`, "info", "ready");
      }
    });
    await capture.start();
    actionLabel.textContent = mode === "enroll" ? "Capture enrollment" : "Verify identity";
    primaryAction.dataset.phase = "capture";
    refreshActionAvailability();
    showResult("Camera ready. Center your face in the guide.", "info");
  }

  async function finishVerification(value) {
    if (value.decision !== "verified") return;
    const resultDetails = await requestJson(`/service-api/sessions/${encodeURIComponent(sessionId)}/result`);
    if (!resultDetails.token) throw new Error("Identity was verified, but the secure result could not be created.");
    const tokenStatus = await requestJson("/service-api/tokens/introspect", {
      method: "POST",
      body: JSON.stringify({ token: resultDetails.token })
    });
    if (!tokenStatus.active) throw new Error("Identity verification could not be completed. Please try again.");
  }

  async function runCapture() {
    const value = mode === "enroll"
      ? await capture.captureEnrollment(true)
      : await capture.captureVerification();
    if (value.decision === "verified") await finishVerification(value);

    if (value.decision === "enrolled") {
      showResult("Your face profile is enrolled.", "success");
    } else if (value.decision === "verified") {
      showResult("Identity verified.", "success");
    } else if (value.decision === "identified") {
      showResult(`Face matched: ${value.identified_subject_id}`, "success");
    } else if (value.decision === "no_match") {
      showResult(value.public_message || "No enrolled face matched.", "info");
    } else if (value.decision === "retry") {
      showResult(value.public_message || "Please adjust your position and try again.", "info");
    } else {
      showResult(value.public_message || "We could not complete this identity check. Please try again.", "error");
      await capture.stop();
      capture = null;
      sessionId = "";
      actionLabel.textContent = "Start camera";
      delete primaryAction.dataset.phase;
    }
    if (value.decision === "enrolled" || value.decision === "verified" || value.decision === "identified" || value.decision === "no_match") {
      await capture.stop();
      capture = null;
      sessionId = "";
      actionLabel.textContent = "Start camera";
      delete primaryAction.dataset.phase;
    }
    if (capture && value.decision === "retry") {
      actionLabel.textContent = mode === "enroll" ? "Capture enrollment" : "Verify identity";
    }
  }

  primaryAction.addEventListener("click", async () => {
    if (processing) return;
    const starting = !capture;
    result.hidden = true;
    setProcessing(true, starting ? "Starting camera…" : mode === "enroll" ? "Capturing…" : "Verifying…");
    try {
      if (starting) {
        await startCamera();
        setCameraStatus("Center your face in the guide.", "info", "idle");
      } else {
        cameraFrame.dataset.state = "busy";
        await runCapture();
      }
    } catch (error) {
      showResult(error.message || "The request could not be completed. Please try again.", "error");
      if (starting && capture) {
        await capture.stop();
        capture = null;
        sessionId = "";
      }
      setCameraStatus(error.message || "Camera setup failed.", "error", "idle");
      actionLabel.textContent = starting ? "Start camera" : mode === "enroll" ? "Capture enrollment" : "Verify identity";
    } finally {
      cameraFrame.dataset.state = capture ? (capture.faceReady ? "ready" : "idle") : "idle";
      setProcessing(false);
    }
  });

  window.addEventListener("pagehide", () => {
    if (capture) capture.stop();
  }, { once: true });
}