## src/api/endpoint.py:
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import List, Dict
import cv2
import numpy as np
from ai.face import FaceProcessor
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from capture.producer import camera_producer
import threading
import logging
import shutil
import base64
import time
from pydantic import BaseModel
import asyncio
import redis.asyncio as redis # <-- MODIFIED: Use async version of redis library
from config.config import settings

import pickle # <-- این را به import ها اضافه کنید
from datetime import datetime, timezone

# --- ADDED: Connection Manager for WebSockets ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
        self.active_connections[camera_id].append(websocket)
        logger.info(f"New WebSocket connection for camera '{camera_id}'. Total connections for this camera: {len(self.active_connections[camera_id])}")

    def disconnect(self, websocket: WebSocket, camera_id: str):
        if camera_id in self.active_connections:
            self.active_connections[camera_id].remove(websocket)
            logger.info(f"WebSocket disconnected for camera '{camera_id}'. Remaining connections: {len(self.active_connections[camera_id])}")

    async def broadcast_to_camera(self, camera_id: str, message: bytes):
        if camera_id in self.active_connections:
            for connection in self.active_connections[camera_id]:
                await connection.send_bytes(message)

manager = ConnectionManager()
# --- END OF ADDED SECTION ---


class RegisterPersonRequest(BaseModel):
    images: List[str]
    personnel_id: str
    first_name: str
    last_name: str

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.debug("Initializing FaceProcessor in main process")
try:
    face_processor = FaceProcessor()
except Exception as e:
    logger.error(f"Failed to initialize FaceProcessor: {e}", exc_info=True)
    raise


camera_threads = {}
stop_events = {}

# --- ADDED: Redis listener for live stream frames ---
# async def redis_frame_listener(manager: ConnectionManager):
#     """Listens to Redis Pub/Sub for new frames and broadcasts them."""
#     redis_client = redis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}", encoding="utf-8", decode_responses=False)
#     pubsub = redis_client.pubsub()
    
#     # Subscribe to all camera output streams
#     await pubsub.psubscribe(f"stream_output_*")
#     logger.info("Redis listener subscribed to 'stream_output_*' channels for live streaming.")
    
#     while True:
#         try:
#             message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
#             if message and message["type"] == "pmessage":
#                 channel_name = message['channel'].decode('utf-8')
#                 camera_id = channel_name.split('_')[-1]
#                 frame_data = message['data']
#                 await manager.broadcast_to_camera(camera_id, frame_data)
#         except Exception as e:
#             logger.error(f"Error in Redis listener: {e}", exc_info=True)
#             await asyncio.sleep(5)

async def redis_frame_listener(manager: ConnectionManager):
    """Listens to Redis Pub/Sub for new frames and broadcasts them."""
    redis_client = redis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}", encoding="utf-8", decode_responses=False)
    pubsub = redis_client.pubsub()
    
    await pubsub.psubscribe(f"stream_output_*")
    logger.info("Redis listener subscribed to 'stream_output_*' channels for live streaming.")
    
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message["type"] == "pmessage":
                # ##<-- بخش اصلاح شده برای محاسبه تأخیر نهایی -->##
                try:
                    payload = pickle.loads(message['data'])
                    frame_data = payload['frame_bytes']
                    processed_ts_str = payload['processed_ts_utc']
                    
                    # محاسبه تأخیر از زمان اتمام پردازش تا رسیدن به API
                    processed_dt = datetime.fromisoformat(processed_ts_str)
                    api_reception_delay = (datetime.now(timezone.utc) - processed_dt).total_seconds()
                    
                    logger.info(f"🚀 Final delivery delay (Process End -> API): {api_reception_delay:.4f}s")

                    channel_name = message['channel'].decode('utf-8')
                    camera_id = channel_name.split('_')[-1]
                    await manager.broadcast_to_camera(camera_id, frame_data)

                except Exception as e:
                     logger.error(f"Could not process published message: {e}")
                # ##<-- پایان بخش اصلاح شده -->##

        except Exception as e:
            logger.error(f"Error in Redis listener: {e}", exc_info=True)
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    """On app startup, start the Redis listener as a background task."""
    logger.info("Application startup: Starting Redis frame listener...")
    asyncio.create_task(redis_frame_listener(manager))
# --- END OF ADDED SECTION ---

def get_iran_timestamp():
    return datetime.now(tz=ZoneInfo("Asia/Tehran")).strftime("%Y%m%d_%H%M%S_%f")

# --- ADDED: WebSocket Endpoint ---
@app.websocket("/ws/stream/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, camera_id)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_id)
# --- END OF ADDED SECTION ---

@app.post("/image")
async def upload_image(file: UploadFile = File(...)):
    # ... (بدون تغییر) ...
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        processed_frame, personnel_ids = face_processor.process_frame(frame, camera_id="image_upload", log_to_db=False)
        if processed_frame is None:
            raise HTTPException(status_code=500, detail="Failed to process image")
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = buffer.tobytes()
        output_dir = "/app/outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = get_iran_timestamp()
        output_path = os.path.join(output_dir, f"processed_{timestamp}.jpg")
        with open(output_path, "wb") as f:
            f.write(processed_image)
        logger.info(f"Processed image saved to {output_path}")
        return {
            "personnel_ids": personnel_ids,
            "processed_image": base64.b64encode(processed_image).decode('utf-8')
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/video")
async def upload_video(file: UploadFile = File(...)):
    # ... (بدون تغییر) ...
    try:
        video_name = file.filename.rsplit(".", 1)[0]
        temp_video_path = f"/tmp/{video_name}_{get_iran_timestamp()}.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        output_dir = f"/app/outputs/videos/{video_name}_{get_iran_timestamp()}"
        os.makedirs(output_dir, exist_ok=True)
        results = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed_frame, personnel_ids = face_processor.process_frame(frame, camera_id="video_upload", log_to_db=False)
            if processed_frame is None: continue
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_image = buffer.tobytes()
            output_path = os.path.join(output_dir, f"frame_{frame_count}_{get_iran_timestamp()}.jpg")
            with open(output_path, "wb") as f:
                f.write(processed_image)
            results.append({
                "frame": frame_count,
                "personnel_ids": personnel_ids,
                "processed_image": base64.b64encode(processed_image).decode('utf-8')
            })
            frame_count += 1
        cap.release()
        os.remove(temp_video_path)
        return {"frames": results}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/rtsp/start")
async def start_rtsp():
    # ... (بدون تغییر) ...
    from config.config import settings
    for camera_id in settings.CAMERAS.keys():
        if camera_id in camera_threads: continue
        stop_events[camera_id] = threading.Event()
        t = threading.Thread(target=camera_producer, args=(camera_id, stop_events[camera_id]))
        t.daemon = True
        camera_threads[camera_id] = t
        t.start()
        time.sleep(1)
        if not t.is_alive():
            del camera_threads[camera_id]
            del stop_events[camera_id]
    return {"message": "RTSP streams started..."}

@app.post("/rtsp/stop")
async def stop_rtsp():
    # ... (بدون تغییر) ...
    for camera_id, stop_event in list(stop_events.items()):
        stop_event.set()
        thread = camera_threads.get(camera_id)
        if thread:
            thread.join(timeout=5)
            del camera_threads[camera_id]
        if camera_id in stop_events:
            del stop_events[camera_id]
    return {"message": "RTSP streams stopped"}

def process_person_in_background(personnel_id: str, saved_image_paths: list, target_dir: str, person_dir_name: str):
    try:
        logger.info(f"BACKGROUND TASK: Starting incremental database update for '{person_dir_name}'.")
        # ##<-- CHANGED: حالا این متد روی نمونه face_processor سرویس API اجرا شده و فایل را ذخیره می‌کند -->##
        face_processor.add_person(personnel_id, saved_image_paths)
        
        # ##<-- REMOVED: دیگر نیازی به ارسال پیام به ردیس نیست -->##
        
        logger.info(f"BACKGROUND TASK: Successfully completed update for '{person_dir_name}'.")

    except ValueError as e:
        shutil.rmtree(target_dir)
        logger.error(f"BACKGROUND TASK FAILED: Could not add person '{person_dir_name}' to database: {e}", exc_info=True)
    except Exception as e:
        shutil.rmtree(target_dir)
        logger.error(f"BACKGROUND TASK FAILED: Unexpected error while adding person '{person_dir_name}': {e}", exc_info=True)

@app.post("/personnel/upload")
async def register_person(data: RegisterPersonRequest, background_tasks: BackgroundTasks):
    DATABASE_ROOT = "/app/src/ai/face_database"
    person_dir_name = f"{data.personnel_id}_{data.first_name}_{data.last_name}"
    target_dir = os.path.join(DATABASE_ROOT, person_dir_name)
    if os.path.exists(target_dir):
        logger.warning(f"Directory for '{person_dir_name}' exists. Updating images.")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    saved_image_paths = []
    try:
        for idx, image_base64 in enumerate(data.images):
            try:
                if "," in image_base64: image_base64 = image_base64.split(",")[1]
                image_data = base64.b64decode(image_base64)
                file_path = os.path.join(target_dir, f"image_{idx+1}.jpg")
                with open(file_path, "wb") as f: f.write(image_data)
                saved_image_paths.append(file_path)
            except Exception as e:
                shutil.rmtree(target_dir)
                raise HTTPException(status_code=500, detail=f"Error saving image {idx+1}: {str(e)}")
    except Exception as e:
        shutil.rmtree(target_dir)
        raise HTTPException(status_code=500, detail="Failed to save uploaded images.")

    background_tasks.add_task(
        process_person_in_background,
        data.personnel_id,
        saved_image_paths,
        target_dir,
        person_dir_name
    )
    return {"message": f"Personnel '{person_dir_name}' accepted for processing."}



## Remove Person
@app.delete("/personnel/delete/{personnel_id}")
async def delete_person_directly(personnel_id: str):
    # ... (بدون تغییر) ...
    logger.info(f"درخواست مستقیم برای حذف شخص با کد پرسنلی: {personnel_id}")
    DB_PICKLE_PATH = "/app/src/ai/face_db.pkl"
    DATABASE_ROOT = "/app/src/ai/face_database"
    pkl_deleted = False
    dir_deleted = False
    person_dir_name = "Not found"
    try:
        if not os.path.exists(DB_PICKLE_PATH):
            raise HTTPException(status_code=500, detail="فایل پایگاه داده (face_db.pkl) پیدا نشد.")
        with open(DB_PICKLE_PATH, "rb") as f: known_faces_data = pickle.load(f)
        if personnel_id not in known_faces_data:
            raise HTTPException(status_code=404, detail=f"شخصی با کد پرسنلی '{personnel_id}' یافت نشد.")
        del known_faces_data[personnel_id]
        with open(DB_PICKLE_PATH, "wb") as f: pickle.dump(known_faces_data, f)
        pkl_deleted = True
        logger.info(f"فایل {DB_PICKLE_PATH} با موفقیت به‌روزرسانی شد.")
        # ##<-- REMOVED: دیگر نیازی به ارسال پیام به ردیس نیست -->##
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطای داخلی سرور در هنگام کار با فایل پایگاه داده: {str(e)}")
    try:
        person_dir_to_delete = None
        for dir_name in os.listdir(DATABASE_ROOT):
            if dir_name.startswith(f"{personnel_id}_"):
                person_dir_to_delete = os.path.join(DATABASE_ROOT, dir_name)
                person_dir_name = dir_name
                break
        if person_dir_to_delete:
            shutil.rmtree(person_dir_to_delete)
            dir_deleted = True
            logger.info(f"پوشه تصاویر '{person_dir_to_delete}' با موفقیت حذف شد.")
        else:
            logger.warning(f"پوشه تصاویر برای کد پرسنلی '{personnel_id}' پیدا نشد.")
    except Exception as e:
        return {
            "message": f"هشدار: اطلاعات شخص با کد '{personnel_id}' از پایگاه داده حذف شد، اما پوشه تصاویر او قابل حذف نبود.",
            "details": {"personnel_id": personnel_id, "deleted_from_pkl": pkl_deleted, "directory_deleted": False, "error": f"خطا در حذف پوشه: {str(e)}"}
        }
    return {
        "message": f"شخص با کد پرسنلی '{personnel_id}' با موفقیت حذف شد.",
        "details": {"personnel_id": personnel_id, "deleted_from_pkl": pkl_deleted, "directory_name": person_dir_name, "directory_deleted": dir_deleted}
    }