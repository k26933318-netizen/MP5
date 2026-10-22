const PREVIEW_WIDTH = 320;
const PREVIEW_QUALITY = 0.6;
const LIVENESS_PREVIEW_WIDTH = 640;
const LIVENESS_PREVIEW_QUALITY = 0.78;
const BURST_QUALITY = 0.9;
const PREVIEW_GAP_MS = 275;
const LIVENESS_PREVIEW_GAP_MS = 100;
const CAPTURE_GAP_MS = 700;
const DEFAULT_BLOCKED_CAMERA_LABELS = ["obs", "virtual", "manycam", "snap camera", "xsplit", "droidcam", "iriun"];
const ENROLL_POSES = [
  ["straight", "Look straight at the camera."],
  ["slightly_left", "Turn slightly to your left."],
  ["slightly_right", "Turn slightly to your right."],
  ["chin_up", "Raise your chin slightly."],
  ["chin_down", "Lower your chin slightly."]
];

export class FaceCapture {
  constructor({
    baseUrl,
    sessionId,
    video,
    onGuidance = () => {},
    onChallenge = () => {},
    onResult = () => {},
    mode = "verify",
    blockedCameraLabels = DEFAULT_BLOCKED_CAMERA_LABELS
  }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.sessionId = sessionId;
    this.video = video;
    this.shapeOverlay = document.querySelector("#face-shape-overlay");
    this.cameraFrame = this.shapeOverlay ? this.shapeOverlay.parentElement : null;
    this.trackFaceBox = this.cameraFrame?.dataset.trackFaceBox === "true"
      && this.cameraFrame.dataset.tracerStyle === "modern";
    this._trackingConfig = JSON.parse(this.cameraFrame?.dataset.trackingConfig || "{}");
    this.fastTrackFaceBox = this.trackFaceBox && this._trackingConfig.mode === "fast";
    // --- face-follow box (added) ---
    this.faceBoxCanvas = null;
    this.faceBoxContext = null;
    this._faceBoxCaptureCanvas = null;
    this._faceBoxCaptureContext = null;
    this._faceBoxTarget = null;
    this._faceBoxRaw = null;
    this._faceBoxFiltered = null;
    this._faceBoxLastRaw = null;
    this._faceBoxDerivative = new Float64Array(4);
    this._faceBoxVelocity = new Float64Array(4);
    this._faceBoxDraw = new Float64Array(4);
    this._faceBoxFilterTime = 0;
    this._faceBoxLastSeen = 0;
    this._faceBoxAlpha = 0;
    this._lastFaceBoxFrameTime = 0;
    this._faceBoxAnimationFrame = null;
    this._faceBoxRequestInFlight = null;
    this._faceBoxLoopPromise = null;
    this._faceBoxLoopActive = false;
    this._faceBoxDeadZone = Number(this._trackingConfig.dead_zone_px) || 2;
    this._faceBoxPadding = Number(this._trackingConfig.padding_percent) || 18;
    this._faceBoxHoldMs = Number(this._trackingConfig.hold_ms) || 400;
    this._faceBoxFadeMs = Number(this._trackingConfig.fade_ms) || 180;
    this._faceBoxDetectionInterval = Number(this._trackingConfig.detection_interval_ms) || 100;
    this._faceBoxMaxPrediction = Number(this._trackingConfig.max_prediction_ms) || 140;
    this._faceBoxMinCutoff = Number(this._trackingConfig.min_cutoff) || 1;
    this._faceBoxBeta = Number(this._trackingConfig.beta) || 0.05;
    this._faceBoxDerivativeCutoff = Number(this._trackingConfig.derivative_cutoff) || 1;
    this._faceBoxFrameWidth = 0;
    this._faceBoxFrameHeight = 0;
    this._faceBoxPixelRatio = window.devicePixelRatio || 1;
    this._faceBoxBracketRatio = 0.18;
    this._faceBoxLineWidth = 2;
    this._faceBoxStrokeStyle = "#82b3df";
    this._faceBoxDebug = null;
    this._faceBoxDebugLastUpdate = 0;
    this._faceBoxDrawFrames = 0;
    this._faceBoxDrawFps = 0;
    this._faceBoxDrawFpsStarted = performance.now();
    this._faceBoxDetectionFps = 0;
    this._faceBoxDetectionCount = 0;
    this._faceBoxDetectionFpsStarted = performance.now();
    this._faceBoxLatency = 0;
    this._faceBoxDetectorMs = 0;
    this._faceBoxLastError = "";
    if (this.fastTrackFaceBox) {
      this.faceBoxCanvas = document.createElement("canvas");
      this.faceBoxCanvas.className = "face-box-tracker";
      this.faceBoxCanvas.setAttribute("aria-hidden", "true");
      this.faceBoxContext = this.faceBoxCanvas.getContext("2d");
      if (!this.faceBoxContext) throw new Error("The browser could not create the face tracer canvas.");
      this.cameraFrame.append(this.faceBoxCanvas);
      this._faceBoxCaptureCanvas = document.createElement("canvas");
      this._faceBoxCaptureContext = this._faceBoxCaptureCanvas.getContext("2d");
      if (!this._faceBoxCaptureContext) throw new Error("The browser could not create the face tracking capture canvas.");
      const frameStyle = getComputedStyle(this.cameraFrame);
      this._faceBoxBracketRatio = Number.parseFloat(frameStyle.getPropertyValue("--tracer-bracket-size")) / 100 || 0.18;
      this._faceBoxLineWidth = Number.parseFloat(frameStyle.getPropertyValue("--tracer-thickness")) || 2;
      this._faceBoxStrokeStyle = frameStyle.getPropertyValue("--tracer-hold").trim() || this._faceBoxStrokeStyle;
      this._resizeFaceBoxCanvas();
      this._faceBoxResizeObserver = new ResizeObserver(() => this._resizeFaceBoxCanvas());
      this._faceBoxResizeObserver.observe(this.cameraFrame);
      if (this._trackingConfig.debug) {
        this._faceBoxDebug = document.createElement("div");
        this._faceBoxDebug.className = "face-box-debug";
        this._faceBoxDebug.setAttribute("aria-hidden", "true");
        this.cameraFrame.append(this._faceBoxDebug);
      }
      this.shapeOverlay.hidden = true;
    }
    this._faceBoxAnimationCallback = (time) => this._drawFaceBoxFrame(time);
    this.onGuidance = onGuidance;
    this.onChallenge = onChallenge;
    this.onResult = onResult;
    this.mode = mode;
    this.blockedCameraLabels = blockedCameraLabels.map((label) => label.toLowerCase());
    this.stream = null;
    this.track = null;
    this._active = false;
    this._previewActive = false;
    this._previewInFlight = null;
    this._previewLoopPromise = null;
    this._baselineSettings = null;
    this._lastPreviewLivenessRequired = false;
    this.faceReady = false;
    this.livenessReady = true;
    this.livenessFailed = false;
    this.fastMode = false;
    this.recognitionFrameCount = 3;
    this.fastCaptureGapMs = 100;
    this.performanceDebug = false;
    this._clientTimings = new Map();
    this._clientTimingSnapshot = {};
    this._clientTimingWindowStarted = performance.now();
    this._clientDeliveredFrames = 0;
    this._clientPreviewFps = 0;
    this._profileTimeline = {
      faceFirstSeenAt: null,
      calibrationDoneAt: null,
      livenessPassedAt: null
    };
    this._recognitionStartedAt = null;
    this._performanceMetadata = {};
    this._latestServerPerformance = {};
    this._videoFrameCallbackId = null;
    this._cameraStartedAt = 0;
    this._capturedFrameCount = 0;
  }

  async start(mode = this.mode) {
    this.mode = mode;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      this.track = this.stream.getVideoTracks()[0];
      await this._rejectVirtualCamera();
      this._baselineSettings = this.track.getSettings();
      this.video.srcObject = this.stream;
      await this.video.play();
      this._cameraStartedAt = performance.now();
      this._capturedFrameCount = 0;
      this._profileTimeline = {
        faceFirstSeenAt: null,
        calibrationDoneAt: null,
        livenessPassedAt: null
      };
      this._clientTimingWindowStarted = this._cameraStartedAt;
      this._clientDeliveredFrames = 0;
      this._active = true;
      this._resumePreview();
      this.onGuidance("Camera ready. Position your face in the frame.");
    } catch (error) {
      if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
      this.track = null;
      this.video.srcObject = null;
      throw error;
    }
  }

  async stop() {
    this._active = false;
    await this._pausePreview();
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.track = null;
    this.video.srcObject = null;
  }

  get faceBox() {
    return this._faceBoxRaw;
  }

  async captureEnrollment(consent = true) {
    if (!this._active || this.mode !== "enroll") throw new Error("Start an enrollment camera session first.");
    if (!this.faceReady) throw new Error("No centered face detected yet. Center your face in the camera guide.");
    await this._pausePreview();
    const frames = [];
    try {
      for (const [, instruction] of ENROLL_POSES) {
        this.onGuidance(instruction);
        await this._delay(CAPTURE_GAP_MS);
        this._assertStableCamera();
        frames.push(this._captureFrame(BURST_QUALITY));
      }
      return await this._submit({ frames, consent });
    } finally {
      this._resumePreview();
    }
  }

  async captureVerification() {
    if (!this._active || !["verify", "identify"].includes(this.mode)) throw new Error("Start a verification camera session first.");
    if (this.livenessFailed) {
      await this._request("/session/liveness/reset", { method: "POST" });
      this.livenessReady = false;
      this.livenessFailed = false;
      return { decision: "retry", public_message: "Look at the camera to try again.", retry_allowed: true };
    }
    if (!this.faceReady) throw new Error("No centered face detected yet. Center your face in the camera guide.");
    const keepPreviewActive = this.fastMode;
    if (!keepPreviewActive) await this._pausePreview();
    try {
      if (this.fastMode) {
        const frames = [];
        for (let index = 0; index < this.recognitionFrameCount; index += 1) {
          if (index > 0) await this._delay(this.fastCaptureGapMs);
          this._assertStableCamera();
          frames.push(this._captureFrame(BURST_QUALITY));
        }
        return await this._submit({ frames });
      }
      const challenge = await this._request("/session/challenge");
      this.onChallenge(challenge);
      this.onGuidance("Hold still for the baseline frame.");
      await this._delay(CAPTURE_GAP_MS);
      const frames = [this._captureFrame(BURST_QUALITY)];
      for (const action of challenge.actions) {
        this.onGuidance(this._challengeInstruction(action));
        for (let index = 0; index < (challenge.frames_per_action || 3); index += 1) {
          await this._delay(CAPTURE_GAP_MS);
          this._assertStableCamera();
          frames.push(this._captureFrame(BURST_QUALITY));
        }
      }
      return await this._submit({ frames, challenge_id: challenge.challenge_id });
    } finally {
      if (!keepPreviewActive) this._resumePreview();
    }
  }

  async _rejectVirtualCamera() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const trackLabel = this.track.label.toLowerCase();
    const selected = devices.find((device) => device.kind === "videoinput" && device.label === this.track.label);
    const label = `${trackLabel} ${selected ? selected.deviceId : ""}`.toLowerCase();
    if (this.blockedCameraLabels.some((term) => label.includes(term))) {
      throw new Error("This camera source is not allowed. Select a physical camera.");
    }
  }

  _resumePreview() {
    if (!this._active || this._previewActive) return;
    this._previewActive = true;
    if (this.fastTrackFaceBox) this._faceBoxAnimationFrame = requestAnimationFrame(this._faceBoxAnimationCallback);
    if (this.fastTrackFaceBox) {
      this._faceBoxLoopActive = true;
      this._faceBoxLoopPromise = this._runFaceBoxLoop();
    }
    this._previewLoopPromise = this._runPreviewLoop();
  }

  async _pausePreview() {
    this._previewActive = false;
    this._faceBoxLoopActive = false;
    if (this._previewInFlight) await this._previewInFlight.catch(() => {});
    if (this._faceBoxRequestInFlight) await this._faceBoxRequestInFlight.catch(() => {});
    if (this._previewLoopPromise) await this._previewLoopPromise;
    if (this._faceBoxLoopPromise) await this._faceBoxLoopPromise;
    this._previewLoopPromise = null;
    this._faceBoxLoopPromise = null;
    if (this._faceBoxAnimationFrame !== null) {
      cancelAnimationFrame(this._faceBoxAnimationFrame);
      this._faceBoxAnimationFrame = null;
    }
    if (this.faceBoxContext) {
      this.faceBoxContext.clearRect(0, 0, this._faceBoxFrameWidth, this._faceBoxFrameHeight);
    }
    this._lastFaceBoxFrameTime = 0;
    if (this.shapeOverlay) {
      this.shapeOverlay.hidden = true;
      this.shapeOverlay.removeAttribute("src");
    }
  }

  async _runPreviewLoop() {
    while (this._active && this._previewActive) {
      const started = performance.now();
      this._previewInFlight = this._sendPreview();
      try {
        const preview = await this._previewInFlight;
        this.faceReady = preview.face_ready;
        this.livenessReady = !preview.liveness_required || Boolean(preview.liveness && preview.liveness.is_live);
        this.livenessFailed = preview.liveness?.state === "failed";
        this._latestServerPerformance = preview.server_performance || {};
        this._recordVerificationPhase(preview);
        if (this.performanceDebug) this._startDeliveredFrameCounter();
        if (!this.fastTrackFaceBox && this.shapeOverlay && Object.hasOwn(preview, "shape_overlay")) {
          this.shapeOverlay.hidden = !preview.shape_overlay;
          if (preview.shape_overlay) this.shapeOverlay.src = preview.shape_overlay;
          else this.shapeOverlay.removeAttribute("src");
        }
        this.onGuidance(preview.guidance, {
          faceReady: this.faceReady,
          livenessRequired: Boolean(preview.liveness_required),
          liveness: preview.liveness || null,
          performance: this._clientTimingSnapshot,
          serverPerformance: preview.server_performance || {},
          performanceMetadata: preview.performance_metadata || {}
        });
      } catch (error) {
        this.faceReady = false;
        this.onGuidance("Face preview is unavailable. Check the connection and try again.", { faceReady: false });
      } finally {
        this._previewInFlight = null;
      }
      if (this._active && this._previewActive) {
        const configuredGap = this._lastPreviewLivenessRequired
          ? LIVENESS_PREVIEW_GAP_MS
          : PREVIEW_GAP_MS;
        await this._delay(Math.max(0, configuredGap - (performance.now() - started)));
      }
    }
  }

  async _sendPreview() {
    const started = performance.now();
    const livenessRequired = this._lastPreviewLivenessRequired;
    const response = await this._request(
      this.fastTrackFaceBox ? "/session/preview" : "/session/preview-overlay",
      {
        method: "POST",
        body: JSON.stringify({
          frame: livenessRequired
            ? this._captureFrame(LIVENESS_PREVIEW_QUALITY, LIVENESS_PREVIEW_WIDTH)
            : this._captureFrame(PREVIEW_QUALITY, PREVIEW_WIDTH)
        })
      }
    );
    this._lastPreviewLivenessRequired = Boolean(response.liveness_required);
    this.fastMode = Boolean(response.liveness_required && response.fast_mode);
    this.performanceDebug = Boolean(response.performance_debug);
    this._performanceMetadata = response.performance_metadata || {};
    if (this.performanceDebug && !this._profileMetadataLogged) {
      console.info("verification_models", {
        ...this._performanceMetadata,
        camera_api: "browser WebRTC / getUserMedia",
        camera_backend: "browser/OS selected backend is not exposed to JavaScript",
        resolution: `${this.video.videoWidth}x${this.video.videoHeight}`
      });
      this._profileMetadataLogged = true;
    }
    this.recognitionFrameCount = Number(response.recognition_frame_count) || 3;
    this.fastCaptureGapMs = Number(response.capture_gap_ms) || 100;
    this._recordClientTiming("preview_roundtrip", performance.now() - started);
    return response;
  }

  async _submit(payload) {
    try {
      this._recognitionStartedAt = performance.now();
      const result = await this._request("/session/submit", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      if (result.decision === "verified" || result.decision === "identified") {
        const verifiedAt = performance.now();
        result.start_to_verified_ms = verifiedAt - this._cameraStartedAt;
        result.frames_between_start_and_verified = this._capturedFrameCount;
        const timeline = this._profileTimeline;
        result.phase_timings_ms = {
          waiting_for_calibration: this._phaseDuration(timeline.faceFirstSeenAt, timeline.calibrationDoneAt),
          waiting_for_liveness_challenge: this._phaseDuration(timeline.calibrationDoneAt, timeline.livenessPassedAt),
          post_liveness_wait_before_submit: this._phaseDuration(timeline.livenessPassedAt, this._recognitionStartedAt),
          recognition_submit_to_verified: verifiedAt - (this._recognitionStartedAt || verifiedAt),
          face_first_seen_to_verified: this._phaseDuration(timeline.faceFirstSeenAt, verifiedAt)
        };
        this._recordClientTiming("verified_total", result.start_to_verified_ms);
        console.info("verification_phase_timings", result.phase_timings_ms);
      }
      this.onResult(result);
      return result;
    } catch (error) {
      const result = {
        decision: "retry",
        public_message: error.message || "The request could not be completed. Please try again.",
        retry_allowed: true
      };
      this.onResult(result);
      return result;
    }
  }

  async _request(path, options = {}) {
    const response = await fetch(`${this.baseUrl}/api/v1${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "X-Session-Id": this.sessionId,
        ...(options.headers || {})
      }
    });
    let body;
    try {
      body = await response.json();
    } catch (error) {
      throw new Error("The service returned an invalid response.");
    }
    const faceBoxHeader = response.headers.get("X-Face-Box");
    if (faceBoxHeader !== null) {
      Object.defineProperty(body, "faceBox", {
        value: faceBoxHeader === "none" ? null : faceBoxHeader.split(",").map(Number),
        enumerable: false
      });
    }
    if (!response.ok) throw new Error(body.public_message || body.message || `The service returned an error (${response.status}).`);
    return body;
  }

  // --- fast smooth tracking (added) ---
  async _runFaceBoxLoop() {
    while (this._active && this._faceBoxLoopActive) {
      const started = performance.now();
      try {
        this._faceBoxRequestInFlight = this._sendFaceBox();
        const result = await this._faceBoxRequestInFlight;
        const detectedAt = performance.now();
        this._faceBoxLatency = detectedAt - started;
        this._faceBoxDetectorMs = Number(result.detector_ms) || 0;
        this._faceBoxLastError = "";
        this._faceBoxDetectionCount += 1;
        if (detectedAt - this._faceBoxDetectionFpsStarted >= 1000) {
          this._faceBoxDetectionFps = this._faceBoxDetectionCount * 1000
            / (detectedAt - this._faceBoxDetectionFpsStarted);
          this._faceBoxDetectionCount = 0;
          this._faceBoxDetectionFpsStarted = detectedAt;
        }
        this._acceptFaceBox(result.face_box, detectedAt);
      } catch (error) {
        this._faceBoxLastError = error.message || "Face tracker request failed.";
        this._acceptFaceBox(null, performance.now());
      } finally {
        this._faceBoxRequestInFlight = null;
      }
      if (this._active && this._faceBoxLoopActive) {
        await this._delay(Math.max(0, this._faceBoxDetectionInterval - (performance.now() - started)));
      }
    }
  }

  async _sendFaceBox() {
    return this._request("/session/track-box", {
      method: "POST",
      body: JSON.stringify({ frame: this._captureFaceBoxFrame() })
    });
  }

  _captureFaceBoxFrame() {
    if (!this.video.videoWidth || !this.video.videoHeight) {
      throw new Error("Camera frame is not ready.");
    }
    const canvas = this._faceBoxCaptureCanvas;
    const width = Number(this._trackingConfig.detector_width) || 320;
    const scale = Math.min(1, width / this.video.videoWidth);
    const canvasWidth = Math.max(1, Math.round(this.video.videoWidth * scale));
    const canvasHeight = Math.max(1, Math.round(this.video.videoHeight * scale));
    if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
    }
    this._faceBoxCaptureContext.drawImage(this.video, 0, 0, canvasWidth, canvasHeight);
    return canvas.toDataURL("image/jpeg", 0.55);
  }

  _acceptFaceBox(faceBox, timestamp) {
    if (!faceBox || faceBox.length !== 4 || faceBox.some((coordinate) => !Number.isFinite(coordinate))) {
      this._faceBoxTarget = null;
      this._faceBoxRaw = null;
      return;
    }
    this._faceBoxRaw = faceBox;
    const videoWidth = this.video.videoWidth;
    const videoHeight = this.video.videoHeight;
    if (!videoWidth || !videoHeight || !this._faceBoxFrameWidth || !this._faceBoxFrameHeight) return;

    const scale = Math.max(this._faceBoxFrameWidth / videoWidth, this._faceBoxFrameHeight / videoHeight);
    const cropX = (videoWidth * scale - this._faceBoxFrameWidth) / 2;
    const cropY = (videoHeight * scale - this._faceBoxFrameHeight) / 2;
    const left = faceBox[0] * videoWidth;
    const top = faceBox[1] * videoHeight;
    const right = faceBox[2] * videoWidth;
    const bottom = faceBox[3] * videoHeight;
    const paddingX = (right - left) * this._faceBoxPadding / 100;
    const paddingY = (bottom - top) * this._faceBoxPadding / 100;
    const target = this._faceBoxTarget || new Float64Array(4);
    target[0] = Math.max(0, left * scale - cropX - paddingX * scale);
    target[1] = Math.max(0, top * scale - cropY - paddingY * scale);
    target[2] = Math.min(this._faceBoxFrameWidth, right * scale - cropX + paddingX * scale) - target[0];
    target[3] = Math.min(this._faceBoxFrameHeight, bottom * scale - cropY + paddingY * scale) - target[1];

    if (this._faceBoxLastSeen && timestamp - this._faceBoxLastSeen > this._faceBoxHoldMs + this._faceBoxFadeMs) {
      this._faceBoxVelocity.fill(0);
      this._faceBoxDerivative.fill(0);
      if (this._faceBoxFiltered) this._faceBoxLastRaw.set(this._faceBoxFiltered);
    }
    const seconds = this._faceBoxFilterTime
      ? Math.min(0.2, Math.max(0.001, (timestamp - this._faceBoxFilterTime) / 1000))
      : 0;
    if (!this._faceBoxFiltered || !seconds) {
      this._faceBoxFiltered = new Float64Array(target);
      this._faceBoxLastRaw = new Float64Array(target);
      this._faceBoxVelocity.fill(0);
      this._faceBoxDerivative.fill(0);
    } else {
      const derivativeAlpha = this._oneEuroAlpha(this._faceBoxDerivativeCutoff, seconds);
      for (let index = 0; index < 4; index += 1) {
        const raw = target[index];
        const previousFiltered = this._faceBoxFiltered[index];
        if (Math.abs(raw - previousFiltered) <= this._faceBoxDeadZone) target[index] = previousFiltered;
        const derivative = (target[index] - this._faceBoxLastRaw[index]) / seconds;
        this._faceBoxDerivative[index] += derivativeAlpha * (derivative - this._faceBoxDerivative[index]);
        const cutoff = this._faceBoxMinCutoff + this._faceBoxBeta * Math.abs(this._faceBoxDerivative[index]);
        const alpha = this._oneEuroAlpha(cutoff, seconds);
        const filtered = previousFiltered + alpha * (target[index] - previousFiltered);
        this._faceBoxVelocity[index] = this._faceBoxVelocity[index] * 0.55
          + ((filtered - previousFiltered) / seconds) * 0.45;
        this._faceBoxFiltered[index] = filtered;
        this._faceBoxLastRaw[index] = target[index];
      }
    }
    this._faceBoxFilterTime = timestamp;
    this._faceBoxLastSeen = timestamp;
    this._faceBoxAlpha = Math.min(1, this._faceBoxAlpha + 0.5);
    this._faceBoxTarget = target;
  }

  _oneEuroAlpha(cutoff, seconds) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / seconds);
  }

  _resizeFaceBoxCanvas() {
    const bounds = this.cameraFrame.getBoundingClientRect();
    this._faceBoxFrameWidth = bounds.width;
    this._faceBoxFrameHeight = bounds.height;
    this._faceBoxPixelRatio = window.devicePixelRatio || 1;
    const width = Math.round(bounds.width * this._faceBoxPixelRatio);
    const height = Math.round(bounds.height * this._faceBoxPixelRatio);
    if (this.faceBoxCanvas.width !== width) this.faceBoxCanvas.width = width;
    if (this.faceBoxCanvas.height !== height) this.faceBoxCanvas.height = height;
    this.faceBoxContext.setTransform(this._faceBoxPixelRatio, 0, 0, this._faceBoxPixelRatio, 0, 0);
  }

  _drawFaceBoxFrame(timestamp) {
    const drawStarted = performance.now();
    this._faceBoxAnimationFrame = null;
    const context = this.faceBoxContext;
    const width = this._faceBoxFrameWidth;
    const height = this._faceBoxFrameHeight;
    context.clearRect(0, 0, width, height);
    const elapsed = this._lastFaceBoxFrameTime ? timestamp - this._lastFaceBoxFrameTime : 16;
    this._lastFaceBoxFrameTime = timestamp;

    if (this._faceBoxFiltered && this._faceBoxAlpha > 0) {
      const age = Math.max(0, timestamp - this._faceBoxLastSeen);
      const predictMs = Math.min(age, this._faceBoxMaxPrediction);
      for (let index = 0; index < 4; index += 1) {
        this._faceBoxDraw[index] = this._faceBoxFiltered[index]
          + this._faceBoxVelocity[index] * predictMs / 1000;
      }
      this._faceBoxDraw[0] = Math.max(0, Math.min(width, this._faceBoxDraw[0]));
      this._faceBoxDraw[1] = Math.max(0, Math.min(height, this._faceBoxDraw[1]));
      this._faceBoxDraw[2] = Math.max(0, Math.min(width - this._faceBoxDraw[0], this._faceBoxDraw[2]));
      this._faceBoxDraw[3] = Math.max(0, Math.min(height - this._faceBoxDraw[1], this._faceBoxDraw[3]));
      if (age > this._faceBoxHoldMs) {
        this._faceBoxAlpha = Math.max(0, 1 - (age - this._faceBoxHoldMs) / this._faceBoxFadeMs);
      } else {
        this._faceBoxAlpha = Math.min(1, this._faceBoxAlpha + elapsed / 150);
      }
      if (this._faceBoxAlpha > 0) {
        const left = this._faceBoxDraw[0];
        const top = this._faceBoxDraw[1];
        const boxWidth = this._faceBoxDraw[2];
        const boxHeight = this._faceBoxDraw[3];
        const bracket = Math.min(boxWidth, boxHeight) * this._faceBoxBracketRatio;
        context.globalAlpha = this._faceBoxAlpha;
        context.strokeStyle = this._faceBoxStrokeStyle;
        context.lineWidth = this._faceBoxLineWidth;
        context.lineCap = "round";
        context.beginPath();
        context.moveTo(left, top + bracket);
        context.lineTo(left, top);
        context.lineTo(left + bracket, top);
        context.moveTo(left + boxWidth - bracket, top);
        context.lineTo(left + boxWidth, top);
        context.lineTo(left + boxWidth, top + bracket);
        context.moveTo(left + boxWidth, top + boxHeight - bracket);
        context.lineTo(left + boxWidth, top + boxHeight);
        context.lineTo(left + boxWidth - bracket, top + boxHeight);
        context.moveTo(left + bracket, top + boxHeight);
        context.lineTo(left, top + boxHeight);
        context.lineTo(left, top + boxHeight - bracket);
        context.stroke();
        context.globalAlpha = 1;
      }
    }

    this._recordClientTiming("drawing", performance.now() - drawStarted);
    this._faceBoxDrawFrames += 1;
    if (timestamp - this._faceBoxDrawFpsStarted >= 1000) {
      this._faceBoxDrawFps = this._faceBoxDrawFrames * 1000
        / (timestamp - this._faceBoxDrawFpsStarted);
      this._faceBoxDrawFrames = 0;
      this._faceBoxDrawFpsStarted = timestamp;
    }
    if (this._faceBoxDebug && timestamp - this._faceBoxDebugLastUpdate >= 250) {
      const error = this._faceBoxLastError ? ` · ${this._faceBoxLastError}` : "";
      this._faceBoxDebug.textContent = `Detect ${this._faceBoxDetectionFps.toFixed(1)} FPS · Draw ${this._faceBoxDrawFps.toFixed(1)} FPS · RTT ${this._faceBoxLatency.toFixed(0)} ms · model ${this._faceBoxDetectorMs.toFixed(0)} ms${error}`;
      this._faceBoxDebugLastUpdate = timestamp;
    }
    if (this._previewActive) {
      this._faceBoxAnimationFrame = requestAnimationFrame(this._faceBoxAnimationCallback);
    }
  }

  _captureFrame(quality, maxWidth = 0) {
    const captureStarted = performance.now();
    if (!this.video.videoWidth || !this.video.videoHeight) throw new Error("Camera frame is not ready.");
    const scale = maxWidth ? Math.min(1, maxWidth / this.video.videoWidth) : 1;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(this.video.videoWidth * scale);
    canvas.height = Math.round(this.video.videoHeight * scale);
    const context = canvas.getContext("2d");
    // The preview can be CSS-mirrored; draw the raw video so turn directions stay unmirrored.
    context.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", quality);
    this._capturedFrameCount += 1;
    this._recordClientTiming("frame_capture", performance.now() - captureStarted);
    return dataUrl;
  }

  _recordClientTiming(stage, durationMs) {
    if (!this.performanceDebug) return;
    const values = this._clientTimings.get(stage) || [];
    values.push(durationMs);
    this._clientTimings.set(stage, values);
    const now = performance.now();
    if (now - this._clientTimingWindowStarted < 3000) return;
    const elapsed = now - this._clientTimingWindowStarted;
    this._clientPreviewFps = this._clientDeliveredFrames * 1000 / elapsed;
    this._clientTimingSnapshot = Object.fromEntries(
      [...this._clientTimings.entries()].map(([name, samples]) => [
        name,
        samples.reduce((sum, value) => sum + value, 0) / samples.length
      ])
    );
    const resolution = `${this.video.videoWidth}x${this.video.videoHeight}`;
    const server = this._latestServerPerformance;
    const profile = {
      camera_fps: Number(this._clientPreviewFps.toFixed(1)),
      resolution,
      capture_ms: this._clientTimingSnapshot.frame_capture ?? null,
      face_detection_ms: server.face_detection ?? null,
      landmarks_ms: server.landmarks ?? null,
      liveness_check_ms: server.liveness ?? null,
      face_embedding_ms: server.face_embedding ?? null,
      matching_ms: server.matching ?? null,
      drawing_ms: this._clientTimingSnapshot.drawing,
      camera_api: "browser WebRTC / getUserMedia",
      metadata: this._performanceMetadata || {},
      frames: this._capturedFrameCount
    };
    console.table([profile]);
    this._clientTimings.clear();
    this._clientDeliveredFrames = 0;
    this._clientTimingWindowStarted = now;
  }

  _startDeliveredFrameCounter() {
    if (this._videoFrameCallbackId !== null || typeof this.video.requestVideoFrameCallback !== "function") return;
    const onVideoFrame = () => {
      if (!this.performanceDebug || !this._active) {
        this._videoFrameCallbackId = null;
        return;
      }
      this._clientDeliveredFrames += 1;
      this._videoFrameCallbackId = this.video.requestVideoFrameCallback(onVideoFrame);
    };
    this._videoFrameCallbackId = this.video.requestVideoFrameCallback(onVideoFrame);
  }

  _recordVerificationPhase(preview) {
    const timeline = this._profileTimeline;
    const now = performance.now();
    const liveness = preview.liveness;
    if (preview.face_ready && timeline.faceFirstSeenAt === null) timeline.faceFirstSeenAt = now;
    if (timeline.calibrationDoneAt === null && liveness?.baseline != null && liveness.instruction !== "Look at the camera") {
      timeline.calibrationDoneAt = now;
    }
    if (liveness?.is_live && timeline.livenessPassedAt === null) timeline.livenessPassedAt = now;
  }

  _phaseDuration(startedAt, endedAt) {
    return startedAt == null || endedAt == null ? null : endedAt - startedAt;
  }

  _assertStableCamera() {
    if (!this.track || this.track.readyState !== "live") throw new Error("Camera stream stopped during capture.");
    const current = this.track.getSettings();
    const baseline = this._baselineSettings || {};
    if ((baseline.width && current.width !== baseline.width) || (baseline.height && current.height !== baseline.height)) {
      throw new Error("Camera resolution changed during capture.");
    }
    if (baseline.frameRate && current.frameRate && Math.abs(current.frameRate - baseline.frameRate) / baseline.frameRate > 0.25) {
      throw new Error("Camera frame rate changed during capture.");
    }
  }

  _challengeInstruction(action) {
    if (action === "blink") return "Blink once, then open your eyes.";
    if (action === "turn_left") return "Turn your head slightly to your left.";
    if (action === "turn_right") return "Turn your head slightly to your right.";
    return "Follow the displayed action.";
  }

  _delay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }
}

export default FaceCapture;