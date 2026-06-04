# AI Reception Monitoring System — Project Summary (for ChatGPT)

Paste this whole file into ChatGPT and ask it to help reduce **lag / latency**.
It describes the full stack, architecture, current settings, and where the lag is.

---

## 1. What the project does
A local, **CPU-only** reception monitor. It takes a live camera stream (USB **or**
IP/RTSP PTZ camera), detects people, recognizes enrolled faces (shows their **name +
department**, otherwise **"Guest"**), tracks each person, estimates posture/activity
(Standing/Sitting/Walking/Running/Bending/Idle), and shows everything on a web
dashboard with department-colored bounding boxes. No cloud, no SQL database.

## 2. Hardware it runs on
- **Laptop, CPU only (no GPU/CUDA used)**
- CPU: Intel Core **i7-11850H** @ 2.5GHz, **16 cores**
- OS: **Ubuntu 24.04 LTS**
- Camera in use now: **IP PTZ camera over RTSP** at `rtsp://192.168.29.199:554/1`
  (native 1920×1080 H.264), connected over LAN/Wi-Fi.

## 3. Tech stack + exact versions
- Language: **Python 3.12.3**
- Web/server: **Flask 3.0.3** (dev server, threaded=True), MJPEG stream over HTTP
- Person detection: **YOLOv8n** via **ultralytics 8.2.103** (device=cpu, imgsz 640)
- Tracking: **ByteTrack** (built into ultralytics) + **lapx 0.9.4**
- Face detection + embeddings: **InsightFace 0.7.3**, model pack **buffalo_sc**,
  backend **onnxruntime 1.18.1** (CPUExecutionProvider)
- Vector search: **faiss-cpu 1.8.0** (IndexFlatIP, cosine similarity)
- Pose/activity: **MediaPipe 0.10.14** (Pose, model_complexity 0)
- Imaging: **opencv-python 4.10.0.84**, **numpy 1.26.4**
- **torch 2.12.0** (CPU, pulled in by ultralytics)
- Storage: **JSON files + FAISS index + .npy embeddings** (no SQL)
- Frontend: server-rendered HTML + vanilla JS + CSS (dark theme), no frameworks

## 4. Pipeline (single shared pipeline, no duplicates)
```
RTSP/USB camera
  → CameraManager (single owner; opens ONE cv2.VideoCapture)
  → Capture thread (keeps ONLY the latest frame, drops old frames = no backlog)
  → Inference thread:
        YOLOv8n person detection (every Nth frame)
        → ByteTrack tracking (stable track IDs)
        → for each track: crop body → InsightFace face → FAISS search → name/department
          (recognized ONCE per track, cached; retried periodically until a clear face)
        → MediaPipe pose (every Mth frame) → activity label
        → draw boxes/labels onto the frame
        → encode JPEG → shared output buffer
  → Display: Flask streams the shared JPEG as MJPEG (multipart/x-mixed-replace)
  → Browser dashboard <img> shows it; same frame used for fullscreen/reception mode
```

### Threading model
- **Capture thread**: reads frames from the camera, always overwrites with the newest
  frame (effective queue size 1). RTSP uses TCP transport + 5s socket timeout.
- **Inference thread**: does all detection/recognition/pose/drawing, then publishes
  one JPEG to a lock-protected buffer.
- **Flask/MJPEG**: streams the latest JPEG to the browser (~30 fps cap on the yield).
- An **FPS limiter** caps the inference loop so it doesn't reprocess identical frames
  faster than the camera delivers them.

## 5. Performance features already implemented
- **FPS limit** (caps processing rate; default 15).
- **Frame skipping** for detection (`detect_every_n_frames`).
- **Pose runs less often** (`pose_every_n_frames`).
- **Recognition caching**: each track is recognized once, name cached on the track ID
  for up to `cache_seconds` (60s); not run every frame.
- **Adaptive low-latency mode**: a CPU monitor (psutil, checked ~1.5s); when CPU > 80%
  it automatically increases frame skip, reduces pose frequency, and lowers JPEG
  quality (in that order) — and recovers when CPU < 60%. It never drops the stream.
- **Latest-frame-only capture** (discards backlog) for low latency.
- Stream/display resolution is decoupled from YOLO input size (imgsz stays 640 for
  speed while the displayed/processed frame can be 720p/1080p).

## 6. Current settings (live values)
```
pipeline.stream_resolution = 1280x720   # processing/display size (frame resized to this)
pipeline.fps_limit         = 15         # max output FPS
pipeline.detect_every_n_frames = 2      # YOLO+ByteTrack runs every 2nd frame
pipeline.pose_every_n_frames   = 5      # MediaPipe pose cadence
pipeline.jpeg_quality      = 70

detection.imgsz       = 640             # YOLOv8n input size (square, letterboxed)
detection.confidence  = 0.4

recognition.threshold        = 0.40     # cosine similarity for a known match
recognition.min_face_size    = 22 px
recognition.recognition_interval = 15   # re-attempt recognition every N frames
recognition.cache_seconds    = 60

tracking.track_buffer = 60              # frames a lost track is kept (ByteTrack)
pose.model_complexity = 0               # fastest MediaPipe pose
```

## 7. THE PROBLEM I need help with: LAGGING / LATENCY
The live video on the dashboard **lags / is delayed / stutters**, especially with the
**RTSP IP PTZ camera** (1080p H.264 over Wi-Fi). I want a smooth, low-latency stream
(target **10–15 FPS, < 300 ms glass-to-glass latency**) on this CPU-only laptop.

### Things that likely contribute to lag (please analyze and give concrete fixes)
1. **RTSP decode cost**: the camera sends 1920×1080 H.264; OpenCV/FFmpeg decodes it on
   CPU, then I resize to 1280×720. Decoding 1080p every frame is heavy.
2. **Network buffering / latency**: RTSP over Wi-Fi + FFmpeg buffering can add delay;
   I set `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|stimeout;5000000` and
   `CAP_PROP_BUFFERSIZE=1`.
3. **MJPEG delivery**: each processed frame is JPEG-encoded (quality 70) and streamed
   to the browser via multipart MJPEG; the browser `<img>` re-renders each frame.
4. **MediaPipe pose** runs on every tracked person’s crop (every 5th frame) — can be a
   CPU spike with multiple people.
5. **InsightFace recognition** retries on unrecognized ("Guest") tracks every 15 frames
   continuously — with several people this adds steady CPU load.
6. **Single inference thread** does detect + track + recognize + pose + draw + encode
   sequentially; a slow stage (pose/recognition) delays the next displayed frame.

### Questions for ChatGPT
- How to **minimize RTSP latency** in OpenCV/FFmpeg on CPU (transport, buffer, flags,
  using the camera’s **sub-stream** `rtsp://192.168.29.199:554/2` at 640×360, hardware
  decode options, or GStreamer pipeline alternatives)?
- Best **settings combo** (stream_resolution, fps_limit, detect_every_n, pose cadence,
  imgsz, jpeg_quality) for smoothest 10–15 FPS on an i7-11850H CPU?
- Should I **decode at a lower resolution** (use the camera sub-stream) and only
  upscale for display? Trade-offs for face recognition accuracy vs latency?
- Would moving **JPEG encoding / MJPEG** to a separate thread, or switching to
  **WebRTC / WebSocket + smaller frames**, reduce perceived lag?
- How to keep the **inference thread from blocking the display** — e.g., decouple a
  dedicated display thread, or run detection/recognition on a worker while the latest
  raw frame is always shown?
- Any **ONNXRuntime / OpenCV / FFmpeg thread or env tuning** (OMP/MKL threads,
  `cv2.setNumThreads`, onnxruntime intra-op threads) that helps on a 16-core CPU?
- Concrete code-level changes to hit **<300 ms latency** while keeping recognition +
  posture working.

Please give me a prioritized, concrete list of changes (settings + code) to reduce the
lag, with the expected impact of each.
