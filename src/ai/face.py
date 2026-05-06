## src/ai/face.py:
import cv2
import numpy as np
import os
import pickle
import logging
import traceback
import time  # <-- وارد کردن کتابخانه زمان
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.cluster import KMeans
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque

# --- Use the generic tracker creator from BoxMOT ---
from boxmot.tracker_zoo import create_tracker

# --- Project-specific Imports ---
from .augment import get_augs, apply_aug
# from .sr import SRWrapper
from database.database import DatabaseLogger

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ai.face")

def l2_normalize(x, eps=1e-10):
    """L2-normalizes a vector."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    return x / (norm + 1e-6) if norm > eps else x

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# --- MODIFIED: ADDED HELPER FUNCTION TO RESIZE FRAME FOR DATABASE ---
def resize_frame_for_database(frame, max_size=640):
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


class TrackedFace:
    """A helper class to store state information for each tracked face."""
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.label = None
        self.recent_labels = deque(maxlen=5)
        self.confirmation_counter = 0
        self.age = 0
        self.frames_since_recognition = 0

class FaceProcessor:
    """
    A stateful FaceProcessor integrating InsightFace with a generic BoxMOT tracker
    for robust, consistent face tracking and recognition.
    """
    def __init__(self,
                 model_path='/app/src/ai',
                 face_db_path='/app/src/ai/face_database',
                 pickle_path='/app/src/ai/face_db.pkl',
                 det_size=(640, 640),
                 device='cpu',
                 metric='euclidean',
                 tracker_type='bytetrack'):

        logger.info(f"Initializing Stateful FaceProcessor (ArcFace + BoxMOT/{tracker_type})")
        self.model_path = model_path
        self.face_db_path = face_db_path
        self.pickle_path = pickle_path
        self.det_size = det_size
        self.device = device
        self.metric = metric.lower() if metric.lower() in ['cosine', 'euclidean'] else 'cosine'
        
        self.detector = FaceAnalysis(name='buffalo_l', root=model_path, allowed_modules=['detection', 'recognition'])
        ctx_id = 0 if device != 'cpu' else -1
        self.detector.prepare(ctx_id=ctx_id, det_size=self.det_size)

        self.match_threshold = 1.0 if self.metric == 'euclidean' else 0.45
        self.ratio_thresh = 0.85
        self.conf_thresh = 0.5

        self.RECOGNITION_INTERVAL = 5
        self.STALE_TRACK_THRESHOLD = 30
        self.CONFIRMATION_COUNT = 3
        self.IOU_MATCHING_THRESHOLD = 0.6

        logger.info(f"Initializing '{tracker_type}' tracker via BoxMOT.")
        self.tracker = create_tracker(tracker_type, tracker_config=os.path.join(model_path, 'models', 'bytetrack.yml'), device=device, half=False, per_class=False)
        self.tracked_faces = {}

        self.face_db = {}
        # ##<-- ADDED: متغیر برای نگهداری زمان آخرین تغییر فایل -->##
        self.pickle_last_modified = 0.0 
        self._load_face_database()
        logger.info("FaceProcessor initialized with %d identities.", len(self.face_db))
        self.db_logger = DatabaseLogger()
        
    def _save_database_to_pickle(self):
        try:
            with open(self.pickle_path, 'wb') as f: pickle.dump(self.face_db, f)
            logger.info("Saved updated face_db pickle to %s with %d identities.", self.pickle_path, len(self.face_db))
            # ##<-- ADDED: بروزرسانی زمان آخرین تغییر پس از ذخیره -->##
            self.pickle_last_modified = os.path.getmtime(self.pickle_path)
        except Exception as e: logger.exception("Failed to save face database pickle: %s", e)

    def _load_face_database(self):
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, 'rb') as f: self.face_db = pickle.load(f)
                # ##<-- ADDED: بروزرسانی زمان آخرین تغییر پس از بارگذاری -->##
                self.pickle_last_modified = os.path.getmtime(self.pickle_path)
                logger.info("Loaded face database from pickle with %d persons.", len(self.face_db))
                return
            except Exception as e: logger.warning("Could not load pickle file: %s. Rebuilding database.", e)
        
        logger.info("Pickle not found. Building new augmented face database...")
        if not os.path.exists(self.face_db_path):
            os.makedirs(self.face_db_path)
            logger.warning("Face database path did not exist, created: %s", self.face_db_path)
            return
            
        aug_pipeline = get_augs(image_size=112)
        temp_person_embs = {}
        person_folders = [p for p in os.listdir(self.face_db_path) if os.path.isdir(os.path.join(self.face_db_path, p))]
        for person_id in person_folders:
            person_dir = os.path.join(self.face_db_path, person_id)
            logger.info(f"Processing DB person: {person_id}")
            person_embeddings = []
            image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in image_files:
                try:
                    pil_img = Image.open(os.path.join(person_dir, img_file)).convert("RGB")
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    faces = self.detector.get(cv_img)
                    if faces:
                        emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                        if emb is not None: person_embeddings.append(emb)
                    for _ in range(40):
                        aug_pil = apply_aug(pil_img, aug_pipeline)
                        aug_cv = cv2.cvtColor(np.array(aug_pil), cv2.COLOR_RGB2BGR)
                        faces = self.detector.get(aug_cv)
                        if faces:
                            emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                            if emb is not None: person_embeddings.append(emb)
                except Exception as e:
                    logger.exception(f"Could not process image {img_file} for {person_id}: {e}")
            if person_embeddings: temp_person_embs[person_id] = np.array(person_embeddings, dtype=np.float32)

        self.face_db = {}
        for pid, embs in temp_person_embs.items():
            if len(embs) == 0: continue
            k = min(5, len(embs))
            if k < 1: continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs)
            prototypes = [l2_normalize(center) for center in kmeans.cluster_centers_]
            self.face_db[pid] = {'prototypes': prototypes, 'n_examples': len(embs)}
            logger.info(f"Built {k} prototypes for {pid} from {len(embs)} examples.")
        self._save_database_to_pickle()

    # ##<-- REPLACED: متد قبلی با یک متد هوشمندتر جایگزین شد -->##
    def check_and_reload_db_if_changed(self):
        """Checks the pickle file's modification time and reloads it if it's newer."""
        try:
            if not os.path.exists(self.pickle_path):
                if self.face_db: # اگر دیتابیس در حافظه وجود داشت اما فایل حذف شده
                    logger.warning("Pickle file seems to be deleted. Clearing in-memory database.")
                    self.face_db = {}
                    self.pickle_last_modified = 0.0
                return

            current_mtime = os.path.getmtime(self.pickle_path)
            
            if current_mtime > self.pickle_last_modified:
                logger.info("Change detected in face_db.pkl. Reloading database...")
                with open(self.pickle_path, 'rb') as f:
                    self.face_db = pickle.load(f)
                self.pickle_last_modified = current_mtime
                logger.info("Successfully reloaded face database. Now tracking %d identities.", len(self.face_db))

        except Exception as e:
            logger.error("Failed to check or reload face database pickle: %s", e, exc_info=True)


    def add_person(self, personnel_id: str, image_paths: list):
        # این متد از سرویس API فراخوانی می‌شود و مستقیماً face_db این نمونه را تغییر می‌دهد
        # و سپس در فایل ذخیره می‌کند که باعث می‌شود زمان mtime آن تغییر کند.
        if personnel_id in self.face_db:
             # جلوگیری از اضافه کردن تکراری در همان نمونه، اگرچه منطق اصلی در API است
            logger.warning(f"Personnel ID '{personnel_id}' already exists in this instance. Overwriting.")

        logger.info(f"Adding new person '{personnel_id}'.")
        aug_pipeline = get_augs(image_size=112)
        person_embeddings = []
        for img_path in image_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                faces = self.detector.get(cv_img)
                if faces:
                    emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                    if emb is not None: person_embeddings.append(emb)
                for _ in range(40):
                    aug_pil = apply_aug(pil_img, aug_pipeline)
                    aug_cv = cv2.cvtColor(np.array(aug_pil), cv2.COLOR_RGB2BGR)
                    faces = self.detector.get(aug_cv)
                    if faces:
                        emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                        if emb is not None: person_embeddings.append(emb)
            except Exception as e: logger.exception(f"Could not process image {img_path} for {personnel_id}: {e}")
        if not person_embeddings: raise ValueError(f"Could not generate face embeddings for {personnel_id}.")
        embs_arr = np.array(person_embeddings, dtype=np.float32)
        k = min(5, len(embs_arr))
        if k < 1: raise ValueError(f"Not enough embeddings for {personnel_id}.")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs_arr)
        prototypes = [l2_normalize(center) for center in kmeans.cluster_centers_]
        self.face_db[personnel_id] = {'prototypes': prototypes, 'n_examples': len(embs_arr)}
        logger.info(f"Successfully added {personnel_id} to the database.")
        self._save_database_to_pickle()
    
    def reload_database(self):
        logger.info("Full database reload requested.")
        if os.path.exists(self.pickle_path):
            try:
                os.remove(self.pickle_path)
                logger.info("Removed existing pickle file to force rebuild.")
            except Exception as e: logger.error(f"Could not remove pickle file for reload: {e}")
        self._load_face_database()

    def _match_one_to_one(self, embeddings_list):
        labels = ["Unknown"] * len(embeddings_list)
        if not embeddings_list or not self.face_db: return labels
        try:
            pids_all = list(self.face_db.keys())
            n_detected, n_identities = len(embeddings_list), len(pids_all)
            if n_identities == 0: return labels
            score_matrix = np.full((n_detected, n_identities), -1.0 if self.metric == 'cosine' else float('inf'), dtype=np.float32)
            for i in range(n_detected):
                emb = embeddings_list[i]
                for j, pid in enumerate(pids_all):
                    # ##<-- ADDED: یک بررسی برای اطمینان از وجود کلید -->##
                    if pid in self.face_db and 'prototypes' in self.face_db[pid]:
                        prototypes = self.face_db[pid]['prototypes']
                        if self.metric == 'cosine': score_matrix[i, j] = np.max(np.dot(prototypes, emb))
                        else: score_matrix[i, j] = np.min(np.linalg.norm(prototypes - emb, axis=1))
            
            if self.metric == 'cosine':
                best_idx = np.argmax(score_matrix, axis=1)
                best_vals = score_matrix[np.arange(n_detected), best_idx]
                score_matrix[np.arange(n_detected), best_idx] = -2.0
                second_best = np.max(score_matrix, axis=1)
                candidates = [(i, best_idx[i], best_vals[i]) for i in range(n_detected) if best_vals[i] >= self.match_threshold and best_vals[i] >= second_best[i] + (1 - self.ratio_thresh)]
                candidates.sort(key=lambda x: x[2], reverse=True)
            else:
                best_idx = np.argmin(score_matrix, axis=1)
                best_vals = score_matrix[np.arange(n_detected), best_idx]
                score_matrix[np.arange(n_detected), best_idx] = float('inf')
                second_best = np.min(score_matrix, axis=1)
                candidates = [(i, best_idx[i], best_vals[i]) for i in range(n_detected) if best_vals[i] <= self.match_threshold and best_vals[i] <= second_best[i] * self.ratio_thresh]
                candidates.sort(key=lambda x: x[2])

            assigned_pids, assigned_dets = set(), set()
            for det_i, pid_j, score in candidates:
                if det_i in assigned_dets: continue
                pid = pids_all[pid_j]
                if pid in assigned_pids: continue
                labels[det_i] = pid
                assigned_pids.add(pid); assigned_dets.add(det_i)
            return labels
        except Exception as e:
            logger.exception("Error in multi-prototype matching: %s", e)
            return ["Unknown"] * len(embeddings_list)

    def process_frame(self, frame, camera_id="default_cam", log_to_db=True):
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to process_frame")
            return frame, []
        try:
            all_detected_faces = self.detector.get(frame)
            high_conf_faces = [face for face in all_detected_faces if face.det_score >= self.conf_thresh] if all_detected_faces else []
            detections_for_tracker = np.array([list(face.bbox) + [face.det_score, 0] for face in high_conf_faces]) if high_conf_faces else np.empty((0, 6))
            online_targets_np = self.tracker.update(detections_for_tracker, frame)
            
            current_track_ids = set()
            embeddings_to_match = []
            ordered_track_ids_for_recog = []

            if online_targets_np.size > 0:
                tracked_bboxes = [target[:4] for target in online_targets_np]
                detected_bboxes = [face.bbox for face in high_conf_faces]

                for i, target in enumerate(online_targets_np):
                    track_bbox = tracked_bboxes[i]
                    track_id = int(target[4])
                    current_track_ids.add(track_id)

                    if track_id not in self.tracked_faces:
                        self.tracked_faces[track_id] = TrackedFace(track_id, track_bbox)
                    
                    track = self.tracked_faces[track_id]
                    track.bbox = track_bbox
                    track.age = 0
                    track.frames_since_recognition += 1

                    if track.label is None or track.frames_since_recognition >= self.RECOGNITION_INTERVAL:
                        ious = [calculate_iou(track_bbox, det_bbox) for det_bbox in detected_bboxes]
                        if not ious: continue
                        
                        best_match_idx = np.argmax(ious)
                        if ious[best_match_idx] >= self.IOU_MATCHING_THRESHOLD:
                            matched_face_obj = high_conf_faces[best_match_idx]
                            if matched_face_obj.normed_embedding is not None:
                                embeddings_to_match.append(matched_face_obj.normed_embedding)
                                ordered_track_ids_for_recog.append(track_id)
            
            if embeddings_to_match:
                raw_labels = self._match_one_to_one(embeddings_to_match)
                
                for i, track_id in enumerate(ordered_track_ids_for_recog):
                    track = self.tracked_faces[track_id]
                    track.frames_since_recognition = 0
                    track.recent_labels.append(raw_labels[i])
                    
                    if len(track.recent_labels) > 0:
                        majority_label = max(set(track.recent_labels), key=list(track.recent_labels).count)
                        
                        if majority_label != "Unknown" and list(track.recent_labels).count(majority_label) >= self.CONFIRMATION_COUNT:
                            if track.label != majority_label:
                                logger.info(f"Track ID {track_id} confirmed as -> {majority_label}")
                            track.label = majority_label

            stale_ids = [track_id for track_id, track in self.tracked_faces.items() if track_id not in current_track_ids]
            for track_id in stale_ids:
                track = self.tracked_faces[track_id]
                track.age += 1
                if track.age > self.STALE_TRACK_THRESHOLD:
                    if track_id in self.tracked_faces:
                        del self.tracked_faces[track_id]

            annotated_frame = frame.copy()
            final_labels_on_frame = []
            recognized_personnel_codes = []

            for track_id, track in self.tracked_faces.items():
                if track_id not in current_track_ids: continue

                x1, y1, x2, y2 = map(int, track.bbox)
                display_label = track.label if track.label else "Processing..."
                final_labels_on_frame.append(display_label)
                
                color = (0, 255, 0) if track.label and track.label != "Unknown" else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID-{track_id}: {display_label}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if track.label and track.label != "Unknown":
                    code = track.label.split('_')[0]
                    if code.isdigit():
                        recognized_personnel_codes.append(code)
            
            # if log_to_db and recognized_personnel_codes:
            #     self.db_logger.log_faces(
            #         camera_id=camera_id,
            #         personnel_ids=list(set(recognized_personnel_codes)),
            #         frame_datetime=datetime.now(tz=ZoneInfo("Asia/Tehran")),
            #         frame_data = None,
            #         memo="0"
            #     )
            
            return annotated_frame, final_labels_on_frame

        except Exception as e:
            logger.exception("Error in stateful process_frame: %s\n%s", e, traceback.format_exc())
            return frame, []
