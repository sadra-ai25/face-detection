# Face Detection & Recognition System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-green) ![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange) ![Docker](https://img.shields.io/badge/Docker-Compose-blue) ![Redis](https://img.shields.io/badge/Redis-Cache-red)

Multi-camera real-time face detection and recognition system for access monitoring. Detects faces from RTSP streams, matches them against a registered face database, and logs recognition events to SQL Server.

## Features

- **Multi-camera RTSP** — simultaneous face monitoring from multiple IP cameras
- **InsightFace recognition** — buffalo_l ONNX model pipeline for high-accuracy face recognition
- **Face database** — register and manage faces via API; database stored as pickle for fast lookup
- **Cosine similarity matching** — configurable threshold for identity matching
- **Unknown face handling** — faces below threshold are flagged as unknown
- **SQL Server sync** — recognition events (person, camera, timestamp, confidence) logged to database
- **Data augmentation** — `augment.py` utility for improving face registration quality
- **REST API** — register new faces, trigger recognition, query logs

## Tech Stack

| Component | Technology |
|---|---|
| Face Detection | InsightFace (buffalo_l) — ONNX |
| Face Recognition | ArcFace embedding via InsightFace |
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
   - Extracts 512-dim embedding per face
   - Compares against face database (cosine similarity)
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
| `GET` | `/health` | Service health check |
| `POST` | `/register` | Register a new face (upload image + person name) |
| `DELETE` | `/unregister/{name}` | Remove a person from the face database |
| `GET` | `/persons` | List all registered persons |
| `POST` | `/recognize` | Run recognition on an uploaded image |
| `GET` | `/logs` | Query recent recognition events |

### Example: Register a Face

```bash
curl -X POST http://localhost/register \
  -F "name=John Doe" \
  -F "file=@/path/to/photo.jpg"
```

### Example: Recognize from Image

```bash
curl -X POST http://localhost/recognize \
  -F "file=@/path/to/frame.jpg"
```

**Response:**

```json
{
  "detections": [
    {
      "person": "John Doe",
      "confidence": 0.93,
      "bbox": [120, 80, 280, 300]
    }
  ]
}
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
