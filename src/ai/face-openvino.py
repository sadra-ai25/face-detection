import cv2
import numpy as np
import os
import pickle
import logging
import traceback
from PIL import Image
from sklearn.cluster import KMeans
from collections import deque, namedtuple
from datetime import datetime
from zoneinfo import ZoneInfo

# --- OpenVINO Integration ---
from openvino.runtime import Core

# --- Use the generic tracker creator from BoxMOT ---
from boxmot.tracker_zoo import create_tracker

# --- Project-specific Imports ---
from .augment import get_augs, apply_aug
from database.database import DatabaseLogger

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ai.face")

AnalyzedFace = namedtuple('AnalyzedFace', ['bbox', 'det_score', 'normed_embedding', 'normed_embedding_is_l2'])
AnalyzedFace.__new__.__defaults__ = (None,) * len(AnalyzedFace._fields)

# در فایل src/ai/face.py این کلاس را جایگزین کنید

class OpenVINOFaceAnalysis:
    """
    A drop-in replacement for InsightFace's FaceAnalysis using OpenVINO for CPU-optimized inference.
    """
    def __init__(self, model_root, det_model_name='detection_model', rec_model_name='recognition_model', device='CPU'):
        logger.info(f"Initializing OpenVINO Face Analysis on device: {device}")
        self.core = Core()
        
        # Load Detection Model
        det_xml_path = os.path.join(model_root, f"{det_model_name}.xml")
        self.compiled_det_model = self._load_and_compile_model(det_xml_path, device)
        self.det_input_layer = self.compiled_det_model.input(0)
        self.det_output_layer = self.compiled_det_model.output(0)
        _, _, self.det_input_h, self.det_input_w = self.det_input_layer.shape
        
        # Load Recognition Model
        rec_xml_path = os.path.join(model_root, f"{rec_model_name}.xml")
        self.compiled_rec_model = self._load_and_compile_model(rec_xml_path, device)
        self.rec_input_layer = self.compiled_rec_model.input(0)
        self.rec_output_layer = self.compiled_rec_model.output(0)
        _, _, self.rec_input_h, self.rec_input_w = self.rec_input_layer.shape
        
        self.det_thresh = 0.4 # آستانه را کمی بالاتر می‌بریم تا از نویز جلوگیری شود

    def _load_and_compile_model(self, xml_path, device_name):
        logger.info(f"Loading model: {xml_path}")
        model = self.core.read_model(xml_path)
        logger.info(f"Compiling model for device: {device_name}")
        compiled_model = self.core.compile_model(model=model, device_name=device_name)
        return compiled_model

    def _preprocess(self, frame, target_h, target_w):
        """
        Prepares a frame for inference: BGR->RGB, resize, transpose, and NORMALIZE.
        """
        # 1. Convert BGR (from OpenCV) to RGB (expected by model)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to the model's exact input dimensions
        resized_frame = cv2.resize(frame_rgb, (target_w, target_h))
        
        # 3. Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        transposed_frame = resized_frame.transpose(2, 0, 1)
        
        # 4. Add the batch dimension to create a 4D tensor (NCHW)
        input_tensor = np.expand_dims(transposed_frame, axis=0)
        
        # ==================== THE DEFINITIVE FIX ====================
        # 5. Convert to float32 and normalize pixel values to the [0.0, 1.0] range.
        # This is the most critical step for model accuracy.
        input_tensor = input_tensor.astype(np.float32) / 255.0
        # ==========================================================
        
        return input_tensor

    def get(self, frame):
        original_h, original_w = frame.shape[:2]
        
        input_tensor_det = self._preprocess(frame, self.det_input_h, self.det_input_w)
        det_results = self.compiled_det_model(input_tensor_det)[self.det_output_layer]

        det_results = np.squeeze(det_results)

        if len(det_results.shape) == 1:
            det_results = np.expand_dims(det_results, axis=0)
            
        logger.debug(f"Detector processed output shape: {det_results.shape}")

        detections = []
        if det_results.shape[0] > 0:
            for det in det_results:
                score = det[2]
                if score > self.det_thresh:
                    x1 = int(det[3] * original_w)
                    y1 = int(det[4] * original_h)
                    x2 = int(det[5] * original_w)
                    y2 = int(det[6] * original_h)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(original_w - 1, x2), min(original_h - 1, y2)

                    if x2 > x1 and y2 > y1:
                        detections.append({'bbox': [x1, y1, x2, y2], 'score': score})
        
        if not detections: return []

        analyzed_faces = []
        for det in detections:
            bbox = det['bbox']
            face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if face_img.size == 0: continue
            
            input_tensor_rec = self._preprocess(face_img, self.rec_input_h, self.rec_input_w)
            embedding = self.compiled_rec_model(input_tensor_rec)[self.rec_output_layer][0]
            normed_embedding = l2_normalize(embedding)
            
            face = AnalyzedFace(bbox=bbox, det_score=det['score'], normed_embedding=normed_embedding)
            analyzed_faces.append(face)

        return analyzed_faces

def l2_normalize(x, eps=1e-10):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    return x / (norm + 1e-6) if norm > eps else x

# ... (بقیه کد فایل face.py بدون هیچ تغییری باقی می‌ماند) ...

def calculate_iou(box1, box2):
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

class TrackedFace:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.label = None
        self.recent_labels = deque(maxlen=5)
        self.confirmation_counter = 0
        self.age = 0
        self.frames_since_recognition = 0

class FaceProcessor:
    def __init__(self,
                 model_path='/app/src/ai',
                 face_db_path='/app/src/ai/face_database',
                 pickle_path='/app/src/ai/face_db.pkl',
                 det_size=(640, 640),
                 device='CPU',
                 metric='euclidean',
                 tracker_type='bytetrack'):

        logger.info(f"Initializing Stateful FaceProcessor (OpenVINO + BoxMOT/{tracker_type})")
        self.model_path = model_path
        self.face_db_path = face_db_path
        self.pickle_path = pickle_path
        self.det_size = det_size
        self.device = device
        self.metric = metric.lower() if metric.lower() in ['cosine', 'euclidean'] else 'cosine'
        
        ir_model_path = os.path.join(self.model_path, "models", "ir_models") 
        self.detector = OpenVINOFaceAnalysis(model_root=ir_model_path, device=self.device)
        
        self.match_threshold = 1.0 if self.metric == 'euclidean' else 0.45
        self.ratio_thresh = 0.85
        self.conf_thresh = self.detector.det_thresh

        self.RECOGNITION_INTERVAL = 5
        self.STALE_TRACK_THRESHOLD = 30
        self.CONFIRMATION_COUNT = 3
        self.IOU_MATCHING_THRESHOLD = 0.6

        logger.info(f"Initializing '{tracker_type}' tracker via BoxMOT.")
        self.tracker = create_tracker(tracker_type, tracker_config=os.path.join(model_path, 'models', 'bytetrack.yml'), device=device, half=False, per_class=False)
        self.tracked_faces = {}

        self.face_db = {}
        self.pickle_last_modified = 0.0 
        self._load_face_database()
        logger.info("FaceProcessor initialized with %d identities.", len(self.face_db))
        self.db_logger = DatabaseLogger()

    def _save_database_to_pickle(self):
        try:
            with open(self.pickle_path, 'wb') as f: pickle.dump(self.face_db, f)
            logger.info("Saved updated face_db pickle to %s with %d identities.", self.pickle_path, len(self.face_db))
            self.pickle_last_modified = os.path.getmtime(self.pickle_path)
        except Exception as e: logger.exception("Failed to save face database pickle: %s", e)

    def _load_face_database(self):
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, 'rb') as f: self.face_db = pickle.load(f)
                self.pickle_last_modified = os.path.getmtime(self.pickle_path)
                logger.info("Loaded face database from pickle with %d persons.", len(self.face_db))
                return
            except Exception as e: logger.warning("Could not load pickle file: %s. Rebuilding database.", e)
        
        logger.info("Pickle not found. Building new augmented face database...")
        if not os.path.exists(self.face_db_path):
            os.makedirs(self.face_db_path)
            logger.warning("Face database path did not exist, created: %s", self.face_db_path)
            return
            
        logger.info("Temporarily creating InsightFace detector to build database...")
        temp_detector = None
        try:
            from insightface.app import FaceAnalysis as InsightFaceAnalysis
            temp_detector = InsightFaceAnalysis(name='buffalo_l', root=self.model_path, allowed_modules=['detection', 'recognition'])
            temp_detector.prepare(ctx_id=-1, det_size=self.det_size)
        except Exception as e:
            logger.error("Could not initialize InsightFace for DB building: %s. Database will be empty.", e)
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
                    faces = temp_detector.get(cv_img)
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
    
    def check_and_reload_db_if_changed(self):
        try:
            if not os.path.exists(self.pickle_path):
                if self.face_db: 
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
            else: # euclidean
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
            high_conf_faces = all_detected_faces if all_detected_faces else []
            detections_for_tracker = np.array([list(face.bbox) + [face.det_score, 0] for face in high_conf_faces]) if high_conf_faces else np.empty((0, 6))
            online_targets_np = self.tracker.update(detections_for_tracker, frame)
            
            current_track_ids = set()
            embeddings_to_match, ordered_track_ids_for_recog = [], []

            if online_targets_np.size > 0:
                tracked_bboxes, detected_bboxes = [t[:4] for t in online_targets_np], [f.bbox for f in high_conf_faces]

                for i, target in enumerate(online_targets_np):
                    track_bbox, track_id = tracked_bboxes[i], int(target[4])
                    current_track_ids.add(track_id)

                    track = self.tracked_faces.setdefault(track_id, TrackedFace(track_id, track_bbox))
                    track.bbox, track.age, track.frames_since_recognition = track_bbox, 0, track.frames_since_recognition + 1

                    if track.label is None or track.frames_since_recognition >= self.RECOGNITION_INTERVAL:
                        ious = [calculate_iou(track_bbox, db) for db in detected_bboxes]
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
                            if track.label != majority_label: logger.info(f"Track ID {track_id} confirmed as -> {majority_label}")
                            track.label = majority_label

            for track_id in list(self.tracked_faces.keys()):
                if track_id not in current_track_ids:
                    self.tracked_faces[track_id].age += 1
                    if self.tracked_faces[track_id].age > self.STALE_TRACK_THRESHOLD:
                        del self.tracked_faces[track_id]

            annotated_frame, final_labels_on_frame, recognized_personnel_codes = frame.copy(), [], []
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
                    if code.isdigit(): recognized_personnel_codes.append(code)
            
            return annotated_frame, final_labels_on_frame

        except Exception as e:
            logger.exception("Error in stateful process_frame: %s", e)
            return frame, []