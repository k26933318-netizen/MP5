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
  const cameraHint = document.querySelector("#camera-hint");
  const backgroundNotice = document.querySelector("#background-notice");
  const noticeStart = document.querySelector("#notice-start");
  const noticeReopen = document.querySelector("#notice-reopen");
  const livenessDebugPanel = document.querySelector("#liveness-debug-panel");
  const result = document.querySelector("#result");
  const resultText = document.querySelector("#result-text");
  let capture = null;
  let sessionId = "";
  let processing = false;
  let showDebugInfo = cameraFrame.dataset.showDebugInfo === "true";
  let verificationSucceeded = false;
  let backgroundCheckTimer = null;
  const noticeSessionKey = backgroundNotice?.dataset.sessionKey || "";
  const whiteBackgroundCanvas = cameraFrame.dataset.checkWhiteBackground === "true"
    ? document.createElement("canvas")
    : null;
  const whiteBackgroundContext = whiteBackgroundCanvas?.getContext("2d", { willReadFrequently: true });
  if (whiteBackgroundCanvas) {
    whiteBackgroundCanvas.width = 160;
    whiteBackgroundCanvas.height = 120;
  }
  if (backgroundNotice) {
    backgroundNotice.hidden = sessionStorage.getItem(noticeSessionKey) === "shown";
  }

  function setCameraStatus(message, kind = "info", state = "idle") {
    cameraStatus.dataset.kind = kind;
    cameraStatusText.textContent = formatCameraStatus(message, kind);
    cameraStatusIcon.textContent = "";
    cameraFrame.dataset.state = state;
    cameraFrame.dataset.tracerState = kind === "error" ? "failed" : state === "ready" ? "locked" : "searching";
  }

  function formatCameraStatus(message, kind) {
    const normalized = message.toLowerCase();
    if (kind === "success" && (normalized.includes("verified") || normalized === "verified")) return "Verified";
    if (["blink twice", "turn your head slightly left", "turn your head slightly right", "open your mouth"].includes(normalized)) return message;
    if (normalized === "move closer" || normalized.includes("move closer") || normalized.includes("too small")) return "Move closer";
    if (normalized === "more light needed") return "More light needed";
    if (normalized === "blink to verify" || normalized.includes("blink") || normalized.includes("verifying") || normalized.includes("hold still") || normalized.includes("centered")) return "Blink to verify";
    if (normalized === "try again" || kind === "error" || normalized.includes("could not verify")) return "Try again";
    return "Looking for face";
  }

  function cameraStatusMessage(message, state, kind) {
    const liveness = state.liveness;
    const normalized = message.toLowerCase();
    if (liveness?.is_live || (kind === "success" && normalized.includes("verified"))) return "Verified";
    if (liveness?.state === "low_quality") return liveness.instruction || "Try again";
    if (liveness?.state === "failed" || kind === "error") return "Try again";
    if (normalized.includes("move closer") || normalized.includes("too small")) return "Move closer";
    if (!state.faceReady) return "Looking for face";
    return liveness?.instruction || "Look at the camera";
  }

  function showResult(message, kind) {
    result.className = `message message--${kind}`;
    resultText.textContent = message;
    result.querySelector(".message-icon").textContent = kind === "success" ? "✓" : kind === "error" ? "!" : "i";
    result.hidden = false;
    if (kind === "success") {
      verificationSucceeded = true;
      if (cameraHint) cameraHint.hidden = true;
    }
    if (kind === "success" || kind === "error") {
      cameraFrame.dataset.tracerState = kind === "success" ? "verified" : "failed";
      cameraStatus.dataset.kind = kind;
      cameraStatusText.textContent = kind === "success" ? "Verified" : "Try again";
      cameraStatusIcon.textContent = "";
    }
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
      primaryAction.disabled = processing || !capture.faceReady || (!capture.livenessReady && !capture.livenessFailed);
      if (capture.livenessFailed) actionLabel.textContent = "Try again";
      else if (actionLabel.textContent === "Try again") actionLabel.textContent = "Verify identity";
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
        const livenessStatus = state.liveness?.status;
        const kind = state.livenessRequired
          ? state.liveness?.is_live
            ? "success"
            : state.liveness?.state === "failed"
              ? "error"
              : "info"
          : state.faceReady ? "success" : "info";
        setCameraStatus(
          state.livenessRequired ? cameraStatusMessage(message, state, kind) : message,
          kind,
          state.faceReady ? "ready" : "idle"
        );
        if (state.liveness?.state === "failed") {
          cameraFrame.dataset.tracerState = "failed";
        }
        if (livenessDebugPanel) {
          const liveness = state.liveness;
          livenessDebugPanel.hidden = !state.livenessRequired || !showDebugInfo || !liveness;
          if (state.livenessRequired && liveness) {
            const clientStages = Object.entries(state.performance || {})
              .map(([name, value]) => `${name} ${Number(value).toFixed(1)} ms`)
              .join(" · ");
            const serverStages = Object.entries(state.serverPerformance || {})
              .map(([name, value]) => name === "fps" ? `${Number(value).toFixed(1)} server FPS` : `${name} ${Number(value).toFixed(1)} ms`)
              .join(" · ");
            livenessDebugPanel.textContent = `Progress ${liveness.progress}/${liveness.required_progress} · EAR ${Number(liveness.ear).toFixed(3)} · Base ${liveness.baseline == null ? "--" : Number(liveness.baseline).toFixed(3)} · Yaw ${Number(liveness.yaw).toFixed(3)} · Mouth ${Number(liveness.mouth).toFixed(3)} · ${Number(liveness.fps).toFixed(1)} FPS · Quality ${Number(liveness.quality_score).toFixed(2)}${clientStages ? ` · ${clientStages}` : ""}${serverStages ? ` · ${serverStages}` : ""}`;
          }
        }
        refreshActionAvailability();
      },
      onChallenge: (challenge) => {
        setCameraStatus("Blink to verify", "info", "ready");
      }
    });
    await capture.start();
    startCameraHint();
    actionLabel.textContent = mode === "enroll" ? "Capture enrollment" : "Verify identity";
    primaryAction.dataset.phase = "capture";
    refreshActionAvailability();
    showResult("Camera ready. Center your face in the guide.", "info");
  }

  function sampleWhiteBackground() {
    if (!whiteBackgroundCanvas || !whiteBackgroundContext || !capture?.video.videoWidth) return null;
    try {
      const width = whiteBackgroundCanvas.width;
      const height = whiteBackgroundCanvas.height;
      const box = capture.faceBox;
      if (!box) return null;
      whiteBackgroundContext.drawImage(capture.video, 0, 0, width, height);
      const pixels = whiteBackgroundContext.getImageData(0, 0, width, height).data;
      const padding = 0.12;
      const left = Math.max(0, (box[0] - padding) * width);
      const top = Math.max(0, (box[1] - padding) * height);
      const right = Math.min(width, (box[2] + padding) * width);
      const bottom = Math.min(height, (box[3] + padding) * height);
      let count = 0;
      let brightnessSum = 0;
      let brightnessSquaredSum = 0;
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          if (x >= left && x <= right && y >= top && y <= bottom) continue;
          const pixel = (y * width + x) * 4;
          const brightness = pixels[pixel] * 0.299 + pixels[pixel + 1] * 0.587 + pixels[pixel + 2] * 0.114;
          brightnessSum += brightness;
          brightnessSquaredSum += brightness * brightness;
          count += 1;
        }
      }
      if (!count) return null;
      const averageBrightness = brightnessSum / count;
      const variation = Math.sqrt(Math.max(
        0,
        brightnessSquaredSum / count - averageBrightness * averageBrightness
      ));
      const minBrightness = Number(cameraFrame.dataset.whiteBackgroundMinBrightness);
      const maxVariation = Number(cameraFrame.dataset.whiteBackgroundMaxVariation);
      return averageBrightness < minBrightness || variation > maxVariation;
    } catch (error) {
      console.error("Could not sample the white-background hint.", error);
      return null;
    }
  }

  function startCameraHint() {
    if (!cameraHint) return;
    verificationSucceeded = false;
    cameraHint.textContent = cameraFrame.dataset.whiteBackgroundRunningHint;
    cameraHint.hidden = false;
    if (cameraFrame.dataset.checkWhiteBackground !== "true" || backgroundCheckTimer !== null) return;
    const interval = Number(cameraFrame.dataset.whiteBackgroundCheckIntervalMs) || 500;
    backgroundCheckTimer = window.setInterval(() => {
      if (verificationSucceeded) {
        stopCameraHint();
        return;
      }
      const needsWhiteBackground = sampleWhiteBackground();
      if (needsWhiteBackground !== null) {
        cameraHint.textContent = needsWhiteBackground
          ? cameraFrame.dataset.whiteBackgroundCheckHint
          : cameraFrame.dataset.whiteBackgroundRunningHint;
      }
    }, interval);
  }

  function stopCameraHint() {
    if (backgroundCheckTimer !== null) {
      window.clearInterval(backgroundCheckTimer);
      backgroundCheckTimer = null;
    }
    if (cameraHint) cameraHint.hidden = true;
  }

  function dismissBackgroundNotice() {
    if (!backgroundNotice) return;
    backgroundNotice.hidden = true;
    sessionStorage.setItem(noticeSessionKey, "shown");
    if (!capture) primaryAction.click();
  }

  noticeStart?.addEventListener("click", dismissBackgroundNotice);
  noticeReopen?.addEventListener("click", () => {
    backgroundNotice.hidden = false;
    noticeStart.focus();
  });

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
    if (capture?.performanceDebug && value.start_to_verified_ms != null && livenessDebugPanel) {
      livenessDebugPanel.textContent += ` · Total ${Number(value.start_to_verified_ms).toFixed(0)} ms · ${Number(value.frames_between_start_and_verified)} captured frames`;
      console.info(`verification_complete total_ms=${Number(value.start_to_verified_ms).toFixed(0)} frames=${Number(value.frames_between_start_and_verified)}`);
    }

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
      stopCameraHint();
      actionLabel.textContent = "Start camera";
      delete primaryAction.dataset.phase;
    }
    if (value.decision === "enrolled" || value.decision === "verified" || value.decision === "identified" || value.decision === "no_match") {
      await capture.stop();
      capture = null;
      sessionId = "";
      stopCameraHint();
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
    if (starting && backgroundNotice && sessionStorage.getItem(noticeSessionKey) !== "shown") {
      backgroundNotice.hidden = false;
      return;
    }
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
        stopCameraHint();
      }
      setCameraStatus(error.message || "Camera setup failed.", "error", "idle");
      actionLabel.textContent = starting ? "Start camera" : mode === "enroll" ? "Capture enrollment" : "Verify identity";
    } finally {
      cameraFrame.dataset.state = capture ? (capture.faceReady ? "ready" : "idle") : "idle";
      setProcessing(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (
      event.key.toLowerCase() !== "d"
      || event.altKey
      || event.ctrlKey
      || event.metaKey
      || event.target instanceof HTMLElement
        && (event.target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName))
    ) {
      return;
    }
    showDebugInfo = !showDebugInfo;
    if (livenessDebugPanel) {
      livenessDebugPanel.hidden = !showDebugInfo || !capture;
    }
  });

  window.addEventListener("pagehide", () => {
    stopCameraHint();
    if (capture) capture.stop();
  }, { once: true });
}