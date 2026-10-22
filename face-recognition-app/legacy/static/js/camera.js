const MAX_CAPTURE_WIDTH = 640;
const CAPTURE_JPEG_QUALITY = 0.8;
const MAX_DETECTION_WIDTH = 320;
const DETECTION_JPEG_QUALITY = 0.6;
const FACE_POLL_INTERVAL_MS = 275;
const MAX_REGISTER_FRAMES = 5;
const MAX_VERIFY_FRAMES = 2;

let capturedImages = [];

async function sendCapturedImage(endpointUrl, extraFormData = {}) {
  const result = document.querySelector("#result");
  result.textContent = "Processing...";
  try {
    const response = await fetch(endpointUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ images: capturedImages, ...extraFormData })
    });
    const data = await readJsonResponse(response);
    if (!response.ok || data.success === false) throw new Error(data.message || "Request failed.");
    result.textContent = data.message || "Done.";
  } catch (error) {
    result.textContent = error.message;
  }
}

async function readJsonResponse(response) {
  const body = await response.text();
  try {
    return JSON.parse(body);
  } catch (error) {
    throw new Error(
      response.status === 404
        ? "API route not found. Restart the Flask server and refresh this page."
        : `Server returned an unexpected response (${response.status}).`
    );
  }
}

function setupCamera({ endpointUrl, captureCount = 1, extraFields = () => ({}) }) {
  const video = document.querySelector("#video");
  const canvas = document.querySelector("#canvas");
  const start = document.querySelector("#start-camera");
  const retake = document.querySelector("#retake");
  const status = document.querySelector("#status");
  const frameLimit = endpointUrl === "/register"
    ? MAX_REGISTER_FRAMES
    : endpointUrl === "/verify"
      ? MAX_VERIFY_FRAMES
      : captureCount;
  const framesToCapture = Math.min(captureCount, frameLimit);
  let pollTimer = null;
  let detectionInFlight = false;
  let stableDetections = 0;
  let flowActive = false;

  function setStatus(message) {
    status.textContent = message;
  }

  function drawCurrentFrame(maxWidth, quality) {
    if (!video.videoWidth || !video.videoHeight) return null;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
  }

  async function checkForFace() {
    if (!flowActive || detectionInFlight) return;
    const image = drawCurrentFrame(MAX_DETECTION_WIDTH, DETECTION_JPEG_QUALITY);
    if (!image) return;
    detectionInFlight = true;
    try {
      const response = await fetch("/detect_face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image })
      });
      const data = await readJsonResponse(response);
      if (!response.ok) throw new Error(data.message || "Face detection failed.");
      if (data.face_detected) {
        stableDetections += 1;
        setStatus("Face detected. Hold still...");
        if (stableDetections >= 3) captureAutomatically();
      } else {
        stableDetections = 0;
        setStatus("Waiting for face...");
      }
    } catch (error) {
      stableDetections = 0;
      setStatus(error.message);
    } finally {
      detectionInFlight = false;
    }
  }

  function stopDetection() {
    flowActive = false;
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function captureAutomatically() {
    stopDetection();
    capturedImages = [];
    for (let index = 0; index < framesToCapture; index += 1) {
      const image = drawCurrentFrame(MAX_CAPTURE_WIDTH, CAPTURE_JPEG_QUALITY);
      if (!image) {
        setStatus("Could not capture a frame. Please try again.");
        return;
      }
      capturedImages.push(image);
      setStatus(`Photo ${index + 1} of ${framesToCapture}`);
      if (index + 1 < framesToCapture) await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
    setStatus("Captured!");
    retake.hidden = false;
    sendCapturedImage(endpointUrl, extraFields());
  }

  function startDetection() {
    stableDetections = 0;
    flowActive = true;
    setStatus("Camera started. Position your face in the frame.");
    pollTimer = window.setInterval(checkForFace, FACE_POLL_INTERVAL_MS);
    checkForFace();
  }

  async function startCamera() {
    stopDetection();
    setStatus("Starting camera...");
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      await video.play();
      start.disabled = true;
      startDetection();
    } catch (error) {
      setStatus(`Camera access failed: ${error.message}`);
    }
  }

  function retakeCapture() {
    capturedImages = [];
    document.querySelector("#result").textContent = "";
    retake.hidden = true;
    startDetection();
  }

  start.addEventListener("click", startCamera);
  retake.addEventListener("click", retakeCapture);
}
