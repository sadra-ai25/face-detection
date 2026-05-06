## src/processing/consumer.py:
import redis
import pickle
import logging
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from ai.face import FaceProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("consumer")

# --- AI anaylsis image max size ---
AI_MAX_SIZE = 1280

def resize_frame_for_ai(frame, max_size):
    """Resizes a frame to have its longest dimension be max_size, preserving aspect ratio."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if max(h, w) <= max_size:
        return frame
        
    if h > w:
        ratio = max_size / h
        new_h = max_size
        new_w = int(w * ratio)
    else:
        ratio = max_size / w
        new_w = max_size
        new_h = int(h * ratio)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def main():
    assigned_camera_id = os.getenv("CAMERA_ID")
    if not assigned_camera_id:
        logger.error("FATAL: CAMERA_ID environment variable not set.")
        sys.exit(1)

    logger.info(f"--- This is the dedicated consumer for camera: [{assigned_camera_id}] ---")

    stream_name = f"camera_stream_{assigned_camera_id}"
    consumer_group = f"processing_group_{assigned_camera_id}"
    consumer_name = f'consumer-{assigned_camera_id}-{os.getpid()}'
    
    logger.info("Initializing FaceProcessor...")
    face_processor = FaceProcessor()
    
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0)

    try:
        redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
    except redis.exceptions.ResponseError:
        logger.info(f"Consumer group '{consumer_group}' already exists.")

    logger.info(f"Consumer '{consumer_name}' starting to read from stream '{stream_name}'...")
    while True:
        try:
            face_processor.check_and_reload_db_if_changed()

            messages = redis_client.xreadgroup(groupname=consumer_group, consumername=consumer_name, streams={stream_name: '>'}, count=1, block=0)

            for _, message_list in messages:
                for message_id, data in message_list:
                    try:
                        payload = pickle.loads(data[b'data'])
                        camera_id = payload['camera_id']
                        frame_id = payload['frame_id']
                        frame_count = payload.get('frame_count', 'N/A')
                        producer_ts_str = payload.get('timestamp_producer_utc')
                        
                        delay = "N/A"
                        if producer_ts_str:
                            producer_dt = datetime.fromisoformat(producer_ts_str)
                            delay_seconds = (datetime.now(timezone.utc) - producer_dt).total_seconds()
                            delay = f"{delay_seconds:.2f}"
                        
                        logger.info(f"🚚 Received task for frame from '{camera_id}' (Count: {frame_count}, ID: {frame_id}). Queue delay: {delay}s")
                        
                        redis_key = f"frame:{frame_id}"
                        frame_bytes = redis_client.get(redis_key)

                        if frame_bytes is None:
                            logger.warning(f"Frame ID {frame_id} not found in Redis (likely expired). Skipping.")
                            redis_client.xack(stream_name, consumer_group, message_id)
                            continue

                        original_frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                        frame_for_ai = resize_frame_for_ai(original_frame, AI_MAX_SIZE)

                        start_time = time.monotonic()
                        
                        annotated_frame, detected_labels = face_processor.process_frame(
                            frame=frame_for_ai,
                            camera_id=camera_id, 
                            log_to_db=True
                        )
                        
                        end_time = time.monotonic()
                        processing_time_ms = (end_time - start_time) * 1000

                        logger.info(f"✔️ AI processing complete for frame (Count: {frame_count}). Found: {detected_labels or 'None'}.")
                        logger.info(f"⏱️ Processing time for frame (Count: {frame_count}): {processing_time_ms:.2f} ms")

                        # ##<-- MODIFIED: بخش ارسال استریم زنده عمومی‌تر شد -->##
                        try:
                            # 1. فریم پردازش شده را به فرمت JPEG انکود کن
                            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            frame_as_bytes = buffer.tobytes()
                            
                            # 2. آن را در یک کانال مشخص در ردیس پابلیش کن
                            channel_name = f'stream_output_{assigned_camera_id}'
                            redis_client.publish(channel_name, frame_as_bytes)
                            logger.debug(f"Published annotated frame to Redis channel '{channel_name}'")
                        except Exception as e:
                            logger.error(f"Failed to publish annotated frame for live stream: {e}")
                        # ##<-- پایان بخش اصلاح شده -->##

                        redis_client.xack(stream_name, consumer_group, message_id)

                    except Exception as e:
                        logger.error(f"Error processing message {message_id.decode()}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            time.sleep(5)

if __name__ == '__main__':
    main()