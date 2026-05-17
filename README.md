# Face Detection & Recognition System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange) ![BoxMOT](https://img.shields.io/badge/BoxMOT-ByteTrack-blue) ![Docker](https://img.shields.io/badge/Docker-Compose-blue) ![Redis](https://img.shields.io/badge/Redis-Cache-red)

Multi-camera real-time face detection and recognition system for access monitoring. Detects faces from RTSP streams, matches them against a registered face database, and logs recognition events to SQL Server.

## Features

- **Multi-camera RTSP** — simultaneous face monitoring from multiple IP cameras
- **InsightFace recognition** — buffalo_l ONNX pipeline; 512-dim ArcFace embeddings with L2 normalization
- **BoxMOT tracker** — ByteTrack algorithm assigns stable track IDs across frames, reducing redundant recognition calls
- **Face database** — register and manage faces via API; `face_db.pkl` for fast in-memory lookup
- **Configurable similarity metric** — cosine or euclidean distance; recent-labels deque + confirmation counter for robust labeling
- **Unknown face handling** — faces below similarity threshold are flagged as unknown
- **WebSocket live streaming** — `/ws/stream/{camera_id}` pushes annotated frames in real time
- **SQL Server sync** — recognition events (person, camera, timestamp, confidence) logged to database
- **Data augmentation** — `augment.py` for improving face registration quality

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | InsightFace (buffalo_l) — ONNX |
| Face Recognition | ArcFace 512-dim embedding + L2 normalization |
| Object Tracker | BoxMOT (ByteTrack) |
| API Server | FastAPI + Uvicorn |
| Frame Queue | Redis |
| Database | Microsoft SQL Server (pyodbc) |
| Containerization | Docker Compose |

## Architecture

```
IP Cameras (RTSP)
      │
      ▼
 Camera Producer (Thread per camera)
   - Captures frames
   - Pushes to Redis queue
      │
      ▼
 Frame Consumer (Thread per camera)
   - Detects all faces in frame (buffalo_l)
   - BoxMOT ByteTrack → stable track IDs
   - Extracts 512-dim ArcFace embedding (L2 normalized)
   - Compares against face_db.pkl (cosine/euclidean)
   - Recent-labels deque + confirmation counter → stable label
   - Labels: known person or "Unknown"
      │
      ├──▶  SQL Server  (recognition log)
      └──▶  FastAPI  (live status & queries)
```

## Prerequisites

- Docker & Docker Compose
- InsightFace buffalo_l ONNX models in `src/ai/models/buffalo_l/`
- Optional: custom IR recognition model at `src/ai/models/ir_models/recognition_model.bin`
- Microsoft SQL Server (reachable from container)

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/sadra-ai25/face-detection.git
cd face-detection

# 2. Configure environment
cp .env.example .env   # edit with your values

# 3. Place InsightFace models
mkdir -p src/ai/models/buffalo_l
# Download buffalo_l from InsightFace model zoo and place .onnx files here

# 4. Start services
docker compose up -d --build
```

## Configuration

| Key | Description | Example |
|---|---|---|
| `SQL_SERVER` | SQL Server address | `192.168.1.100\sqlserver` |
| `SQL_DATABASE` | Database name | `FaceDB` |
| `SQL_UID` | SQL login | `sa` |
| `SQL_PWD` | SQL password | `your_password_here` |
| `CAMERAS` | JSON map of camera IDs to RTSP URLs | `{"cam1": {"rtsp": "rtsp://..."}}` |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/image` | Run face recognition on a single uploaded image |
| `POST` | `/video` | Upload a video file for batch face recognition |
| `POST` | `/rtsp/start` | Start RTSP stream processing |
| `POST` | `/rtsp/stop` | Stop RTSP stream processing |
| `POST` | `/personnel/upload` | Register a person (base64 images + ID + name) |
| `DELETE` | `/personnel/delete/{personnel_id}` | Remove a person from the face database |
| `WS` | `/ws/stream/{camera_id}` | WebSocket for live annotated frame stream |

### Example: Register a Person

```bash
curl -X POST http://localhost:8000/personnel/upload \
  -H "Content-Type: application/json" \
  -d '{
    "personnel_id": "EMP001",
    "first_name": "Ali",
    "last_name": "Rezaei",
    "images": ["<base64_image_1>", "<base64_image_2>"]
  }'
```

### Example: Recognize from Image

```bash
curl -X POST http://localhost:8000/image \
  -F "file=@/path/to/frame.jpg"
```

### Example: Live Stream via WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/stream/cam1");
ws.onmessage = (event) => {
  const blob = new Blob([event.data], { type: "image/jpeg" });
  // render annotated frame
};
```

## Face Database Management

```bash
# Augment training images for a person
python augment.py --input ./photos/john --output ./augmented/john

# Inspect face database contents
python check_pkl.py

# Modify or merge face database entries
python modify_pkl.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## License

MIT
