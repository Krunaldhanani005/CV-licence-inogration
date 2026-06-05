# AI Reception System — Settings Reference (for ChatGPT)

Paste this whole file into ChatGPT. It explains every tunable setting, the CURRENT
value, what it does, and which direction to change it. Then ask ChatGPT to suggest
values for a goal (smoother / sharper / more accurate recognition / less identity
mix-up, etc.).

---

## How settings work (IMPORTANT)
- Two files hold settings:
  - `config/default.yaml` — base defaults.
  - `data/configs/runtime.json` — **runtime overrides; these WIN over defaults.**
    The Admin Panel "System Settings" page writes here. Editing this file is the
    way to change behavior.
- After editing `runtime.json`, **restart the app** (`Ctrl+C`, then `python app.py`)
  OR change values in the **System Settings** web page and click Save (auto-applies).
- Pipeline = single thread: Camera → YOLOv8n detect → ByteTrack track → (async thread)
  InsightFace face → FAISS match → identity → motion-based activity → draw → MJPEG.

## Hardware / current setup
- CPU-only laptop (no GPU). Python 3.12.
- Camera: RTSP PTZ "ALLBOTIX PTZ" @ `rtsp://192.168.29.199:554/1` (1080p H.264).
- Models: YOLOv8n (person), ByteTrack (tracking), InsightFace buffalo_sc (faces),
  FAISS (match). Activity is **motion-based** (no MediaPipe per person).

=================================================================
## CURRENT EFFECTIVE SETTINGS (from runtime.json) + what each does
=================================================================

### pipeline  (stream + performance)
| Setting | Current | What it does | Change direction |
|---|---|---|---|
| stream_resolution | 1920x1080 | Resolution the frame is processed/displayed at. Higher = sharper + more far-detail, but heavier CPU. | Lower to 1280x720 for big FPS gain; 1920x1080 = sharpest. |
| fps_limit | 15 | Max output FPS (caps processing rate). | 12–15 good on CPU. Lower = smoother+more latency budget. |
| detect_every_n_frames | 1 | Run YOLO+tracker every N frames. 1 = every frame (tight boxes, heaviest). | 2 = ~half CPU, boxes still smooth (box_smooth_alpha hides lag). |
| pose_every_n_frames | 10 | (legacy) activity re-eval cadence; activity is cheap motion math now. | Leave. |
| jpeg_quality | 60 | MJPEG stream quality 1–100. Lower = less bandwidth/CPU, softer image. | 60–80. |
| max_missed_frames | 10 | Frames a track may be unseen before its box is removed (~0.7s). | Lower = boxes vanish faster; higher = more ghost tolerance. |
| box_smooth_alpha | 0.8 | Bounding-box motion smoothing. High = box snaps to person (low lag). | 0.6–0.9. Lower = smoother but laggier box. |
| async_recognition | true | Run face recognition on its own thread (no stream stutter). | Keep true. |

### activity (Standing / Walking / Idle — motion based)
| Setting | Current | What it does | Change direction |
|---|---|---|---|
| walk_motion_threshold | 0.018 | Motion above this = Walking. | Lower (0.014) catches slower walkers; higher = stricter. |
| idle_motion_threshold | 0.005 | Motion below this counts as still → Idle. | Higher = easier Idle. |
| activity_smooth_window | 5 | Majority-vote window for the label (frames). | Higher (9) = calmer/less flicker, slower to change. |

> NOTE: This system version uses simple motion thresholds + a majority vote. (An
> upgraded version adds EMA + hysteresis + box-scale motion to kill flicker and
> catch walking toward/away from the camera — ask ChatGPT for that if labels flicker.)

### detection (YOLOv8n person detector)
| Setting | Current | What it does | Change direction |
|---|---|---|---|
| imgsz | 480 | YOLO input size. Higher = detects SMALL/FAR people better, more CPU. | 640/768 to catch far people (current 480 is low → misses distant people). |
| confidence | 0.45 | Min detection confidence. | Lower (0.35) catches faint/far people but more false boxes. |
| min_confirm_frames | 3 | A track must be seen this many times before its box shows (kills flicker boxes). | 2–4. |

### tracking (ByteTrack)  — set in config/bytetrack.yaml + here
| Setting | Current | What it does | Change direction |
|---|---|---|---|
| track_buffer | 15 | Frames a lost track is kept for re-association (~1s). Low = less id reuse, fast box removal; high = survives occlusion but more id-reuse/transfer risk. | 10–20. |

### recognition (InsightFace + FAISS + identity binding)
| Setting | Current | What it does | Change direction |
|---|---|---|---|
| threshold | 0.45 | Cosine similarity needed to accept a known face. **The guard against wrong names.** | Higher (0.5) = stricter, more "Guest". Lower (0.4) = more matches, more risk. |
| det_size | 960 | Face-detector input size (on the recognition thread). Higher = detects FAR faces. | 640 lighter; 960/1280 better far faces (no stream impact, it's async). |
| min_face_size | 12 | Smallest face (px) to attempt recognition. | 20–30 safer (12 is very small → unreliable matches). |
| min_det_score | 0.40 | Face-detector confidence floor (quality gate). | Higher = stricter face quality. |
| min_sharpness | 8.0 | Blur gate (Laplacian variance); rejects blurry faces. | Higher = reject more blurry faces. |
| recognition_interval | 12 | How often (frames) the face scan runs. | Higher = less CPU, slower first-recognition. |
| revalidation_interval | 45 | Re-verify a named track every N frames. | — |
| cache_seconds | 30 | Reuse a track's identity for this long. | — |
| reverify_gap_frames | 5 | If a named track is lost this many frames then reappears, treat as possible NEW person → clear identity + re-verify. Prevents id-reuse transfer. | Lower = stricter (more re-verify); higher = more persistence. |
| appearance_corr_threshold | 0.40 | Body-clothing color match. Below this = different body (caught a track swap during crossing). | Higher (0.5)=stricter swap detection; lower (0.3)=fewer false resets. |
| appearance_mismatch_frames | 4 | Sustained mismatch frames before clearing identity on a swap. | Lower = faster swap correction; higher = fewer false resets. |
| recognize_max_attempts | 2 | (legacy) attempts before locking Guest. | — |
| debug | false | Verbose `[ID]` per-track logs (track create/lost/assign/invalidate). | Set true to debug identity issues, watch logs/system.log. |

### pose
| Setting | Current | What it does |
|---|---|---|
| enabled | true | Activity labels on/off. |
| model_complexity | 0 | (legacy) 0 = fastest. |

=================================================================
## COMMON GOALS → WHICH SETTINGS TO CHANGE
=================================================================

- **Smoother stream / less lag (CPU):** stream_resolution 1280x720, detect_every_n_frames 2,
  imgsz 640, jpeg_quality 60. (Biggest lever = resolution.)
- **Sharper picture:** stream_resolution 1920x1080, jpeg_quality 80. (Costs FPS.)
- **Detect far / small people:** imgsz 640→768, confidence 0.45→0.40.
- **Recognize far faces:** det_size 960 (keep), min_face_size 20, threshold 0.45.
- **Fewer wrong names (prefer Guest):** threshold 0.48–0.50, min_face_size 24, min_sharpness 12.
- **Stop identity transfer between people:** reverify_gap_frames 3–4, appearance_corr_threshold 0.5,
  appearance_mismatch_frames 3, track_buffer 12.
- **Boxes lag behind walkers:** detect_every_n_frames 1, box_smooth_alpha 0.85.
- **Activity flickers (Standing↔Walking):** activity_smooth_window 9, walk_motion_threshold 0.02.

## Rules / trade-offs to tell ChatGPT
- CPU-only: every "higher quality" setting costs FPS. Target stable 12–15 FPS.
- threshold is the safety against wrong identity — don't drop below 0.42.
- imgsz drives small-person DETECTION; det_size drives far-face RECOGNITION.
- stream_resolution drives picture clarity AND CPU the most.
- After any change: restart the app (or Save in System Settings page).
