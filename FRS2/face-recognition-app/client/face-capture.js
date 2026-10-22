const PREVIEW_WIDTH = 320;
const PREVIEW_QUALITY = 0.6;
const BURST_QUALITY = 0.9;
const PREVIEW_GAP_MS = 275;
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
    this.faceReady = false;
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
    if (!this.faceReady) throw new Error("No centered face detected yet. Center your face in the camera guide.");
    await this._pausePreview();
    try {
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
      this._resumePreview();
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
    this._previewLoopPromise = this._runPreviewLoop();
  }

  async _pausePreview() {
    this._previewActive = false;
    if (this._previewInFlight) await this._previewInFlight.catch(() => {});
    if (this._previewLoopPromise) await this._previewLoopPromise;
    this._previewLoopPromise = null;
  }

  async _runPreviewLoop() {
    while (this._active && this._previewActive) {
      const started = performance.now();
      this._previewInFlight = this._sendPreview();
      try {
        const preview = await this._previewInFlight;
        this.faceReady = preview.face_ready;
        this.onGuidance(preview.guidance, { faceReady: preview.face_ready });
      } catch (error) {
        this.faceReady = false;
        this.onGuidance("Face preview is unavailable. Check the connection and try again.", { faceReady: false });
      } finally {
        this._previewInFlight = null;
      }
      if (this._active && this._previewActive) {
        await this._delay(Math.max(PREVIEW_GAP_MS, PREVIEW_GAP_MS - (performance.now() - started)));
      }
    }
  }

  async _sendPreview() {
    const response = await this._request("/session/preview", {
      method: "POST",
      body: JSON.stringify({ frame: this._captureFrame(PREVIEW_QUALITY, PREVIEW_WIDTH) })
    });
    return response;
  }

  async _submit(payload) {
    try {
      const result = await this._request("/session/submit", {
        method: "POST",
        body: JSON.stringify(payload)
      });
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
    if (!response.ok) throw new Error(body.public_message || body.message || `The service returned an error (${response.status}).`);
    return body;
  }

  _captureFrame(quality, maxWidth = 0) {
    if (!this.video.videoWidth || !this.video.videoHeight) throw new Error("Camera frame is not ready.");
    const scale = maxWidth ? Math.min(1, maxWidth / this.video.videoWidth) : 1;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(this.video.videoWidth * scale);
    canvas.height = Math.round(this.video.videoHeight * scale);
    const context = canvas.getContext("2d");
    // The preview can be CSS-mirrored; draw the raw video so turn directions stay unmirrored.
    context.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
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