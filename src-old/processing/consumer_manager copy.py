import subprocess
import time
import sys
import os
import logging
from config.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConsumerManager")

def main():
    logger.info("🚀 Starting Consumer Manager...")
    
    try:
        camera_ids = list(settings.CAMERAS.keys())
        if not camera_ids:
            logger.error("⚠️ No cameras found in the CAMERAS configuration in .env file. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to parse CAMERAS from .env file. Error: {e}")
        sys.exit(1)

    logger.info(f"✅ Found {len(camera_ids)} cameras to manage: {camera_ids}")

    processes = {}

    while True:
        for cam_id in camera_ids:
            if cam_id not in processes or processes[cam_id].poll() is not None:
                if cam_id in processes:
                    logger.warning(f"🔄 Consumer for camera '{cam_id}' has terminated. Restarting...")
                else:
                    logger.info(f"✨ Starting new consumer for camera '{cam_id}'...")

                env = os.environ.copy()
                env["CAMERA_ID"] = cam_id

                cmd = [sys.executable, "src/processing/consumer.py"]

                try:
                    proc = subprocess.Popen(cmd, env=env)
                    processes[cam_id] = proc
                    logger.info(f"👍 Consumer for '{cam_id}' started successfully with PID {proc.pid}.")
                except Exception as e:
                    logger.error(f"🔥 Failed to start consumer for '{cam_id}'. Error: {e}")
        
        time.sleep(30)

if __name__ == "__main__":
    main()