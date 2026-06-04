# AI Reception Monitoring System

A production-ready, **CPU-only** reception monitoring system that detects people on
an RTSP/IP (PTZ) camera stream, recognises known faces, tracks each person, and
shows their **name** (or **Guest**) plus **posture/activity** on a live reception
monitor — all running locally with no SQL database.

```
RTSP Stream → YOLOv8n Person Detection → ByteTrack Tracking
           → Face Detection → InsightFace Embedding → FAISS Search → Name
           → MediaPipe Pose → Full-Body Box on Live Screen
```

## Features

- **Person detection** — YOLOv8n, person class only, full-body boxes, CPU optimised.
- **Tracking** — ByteTrack; face recognition runs **once per tracked person**, cached
  until the track is lost, then re-recognised.
- **Face recognition** — InsightFace embeddings + FAISS cosine search. Multiple
  photos per person, averaged embedding at enrollment. Metadata in JSON (no SQL).
- **Face→body association** — face found inside the body box; name drawn on the
  full-body box only (never a separate face box).
- **Posture/activity** — MediaPipe Pose: Standing, Sitting, Walking, Running,
  Bending, Idle.
- **Unknown handling** — below threshold ⇒ label is exactly **Guest** (never
  "Unknown/Visitor/Employee").
- **Threaded pipeline** — separate capture / inference / display, frame-skipping.
- **Admin Panel** — Dashboard, Live Monitoring, Camera Settings, Person Management,
  System Settings. Modern responsive dark theme.

## Project structure

```
cv_model/
├── app.py                  # Flask entrypoint (loads models, starts pipeline)
├── requirements.txt
├── config/
│   ├── default.yaml        # base configuration
│   ├── bytetrack.yaml      # tracker tuning
│   └── settings.py         # config manager (YAML + runtime JSON overrides)
├── core/
│   ├── camera/             # threaded RTSP/webcam capture + connection test
│   ├── detection/          # YOLOv8n person detector
│   ├── tracking/           # ByteTrack wrapper
│   ├── recognition/        # InsightFace + FAISS + JSON face DB
│   ├── pose/               # MediaPipe pose/activity estimator
│   └── utils/              # logging + geometry helpers
├── services/               # pipeline orchestration + person enrollment service
├── routes/                 # Flask blueprints (pages + JSON/stream API)
├── templates/  static/     # dark-theme UI
├── data/                   # faces/ embeddings/ faiss/ configs/  (auto-created)
├── models/                 # downloaded model weights cached here
└── logs/
```

## Setup (CPU-only laptop)

Requires Python 3.10 or 3.11.

```bash
cd cv_model
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

First run downloads the YOLOv8n weights and the InsightFace `buffalo_sc` pack
automatically (internet required once; cached under `models/`).

## Run

```bash
python app.py
```

Open **http://localhost:5000** on the reception monitor.

> If no camera is configured, the system falls back to the local webcam (index 0)
> so you can test immediately.

## Configure the camera

Go to **Camera Settings** and either:

- paste a full **RTSP URL**, or
- enter **IP / Username / Password / Channel** (a Hikvision/ONVIF-style URL is
  built automatically).

Click **Test Connection**, then **Save Configuration** (the pipeline restarts).

## Enroll people

**Person Management → Add Person** → enter a name and upload one or more clear
face photos. On save the system:

1. stores photos under `data/faces/<id>/`,
2. computes an InsightFace embedding per photo and **averages** them,
3. saves the embedding to `data/embeddings/<id>.npy`,
4. **rebuilds the FAISS index** automatically.

Recognised people show their name; anyone below the threshold shows **Guest**.

## Tuning (System Settings)

| Setting | Effect |
|---|---|
| Recognition Threshold | Higher = stricter matching (more Guests) |
| Detect Every N Frames | Higher = less CPU, less responsive boxes |
| Pose Every N Frames | Higher = less CPU for activity detection |
| imgsz / Detection Confidence | Speed vs accuracy trade-off |
| Pose model complexity | 0 = fastest on CPU |

## Storage

JSON + FAISS + local files only — **no SQL database**.

- `data/configs/people.json` — person metadata
- `data/embeddings/*.npy` — averaged embeddings
- `data/faiss/faces.index` — FAISS index
- `data/configs/runtime.json` — admin-panel overrides

## Notes

- All inference uses `CPUExecutionProvider` / `device=cpu`.
- The Flask dev server is fine for a single reception monitor. For multiple
  viewers put it behind a production WSGI server (e.g. `waitress`/`gunicorn`)
  with a single worker (the pipeline holds in-process model state).
# CV-licence-inogration
