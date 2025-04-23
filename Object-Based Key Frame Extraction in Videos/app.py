from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import torch
import cv2
import numpy as np
import gc
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_FOLDER, 'keyframes'), exist_ok=True)

model = None

def load_model():
    global model
    if model is None:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects(frames):
    with torch.no_grad():
        results = model(frames)
    return results

def calculate_saliency(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    intensity_contrast = np.std(gray)
    color_contrast = np.std(frame)
    saliency_score = intensity_contrast + color_contrast
    return saliency_score

def select_keyframes(frames, saliency_scores, threshold=0.5):
    keyframes = []
    keyframe_indices = []
    for i, score in enumerate(saliency_scores):
        if score > threshold:
            keyframes.append(frames[i])
            keyframe_indices.append(i)
    return keyframes, keyframe_indices

def detect_scene_transitions(frames, threshold=0.3):
    transitions = []
    for i in range(1, len(frames)):
        hist1 = cv2.calcHist([frames[i-1]], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frames[i]], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        if similarity < threshold:
            transitions.append(i)
    return transitions

def generate_summary(keyframes, output_path="summary.mp4", fps=30):
    if len(keyframes) > 0:
        height, width, _ = keyframes[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in keyframes:
            video.write(frame)
        video.release()
        print(f"Summary video saved to {output_path}")
        return True
    else:
        print("No keyframes selected, cannot generate summary.")
        return False

def process_video(video_path, session_id, frame_resize=(320, 240), frame_sample_rate=5, batch_size=16):
    global model
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, []
    summary_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{session_id}_summary.mp4")
    keyframes_dir = os.path.join(app.config['PROCESSED_FOLDER'], 'keyframes', session_id)
    os.makedirs(keyframes_dir, exist_ok=True)

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_sample_rate == 0:
            frame = cv2.resize(frame, frame_resize)
            frames.append(frame)
        frame_count += 1
    cap.release()
    print(f"Read {len(frames)} frames from video")

    labeled_frames = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        results = detect_objects(batch)
        for frame_idx, frame in enumerate(batch):
            result = results.pandas().xyxy[frame_idx]
            frame_copy = frame.copy()
            for _, row in result.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                confidence, class_name = row['confidence'], row['name']
                if confidence > 0.5:
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            labeled_frames.append(frame_copy)
        del results, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"Processed {len(labeled_frames)} frames with object detection")

    saliency_scores = [calculate_saliency(frame) for frame in labeled_frames]
    keyframes, keyframe_indices = select_keyframes(
        labeled_frames, saliency_scores,
        threshold=np.mean(saliency_scores) * 0.9
    )

    transition_frames = detect_scene_transitions(labeled_frames)
    transition_keyframes = [labeled_frames[i] for i in transition_frames if i < len(labeled_frames)]

    print(f"Selected {len(keyframes)} keyframes and {len(transition_keyframes)} transition frames")

    keyframe_paths = []
    for i, frame in enumerate(keyframes):
        keyframe_subpath = f"keyframes/{session_id}/keyframe_{i}.jpg"
        abs_keyframe_path = os.path.join(app.config['PROCESSED_FOLDER'], keyframe_subpath)
        os.makedirs(os.path.dirname(abs_keyframe_path), exist_ok=True)
        success = cv2.imwrite(abs_keyframe_path, frame)
        if success:
            keyframe_paths.append(keyframe_subpath)

    all_keyframes = keyframes + transition_keyframes
    if all_keyframes and generate_summary(all_keyframes, output_path=summary_path, fps=30):
        return f"{session_id}_summary.mp4", keyframe_paths
    else:
        return None, keyframe_paths

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        frame_resize = (
            int(request.form.get('width', 320)),
            int(request.form.get('height', 240))
        )
        frame_sample_rate = int(request.form.get('sample_rate', 5))
        batch_size = int(request.form.get('batch_size', 16))
        summary_file, keyframe_paths = process_video(
            file_path,
            session_id,
            frame_resize=frame_resize,
            frame_sample_rate=frame_sample_rate,
            batch_size=batch_size
        )
        os.remove(file_path)
        print("Keyframe paths:", keyframe_paths)
        return render_template(
            'results.html',
            summary_file=summary_file,
            keyframe_paths=keyframe_paths,
            session_id=session_id
        )
    return redirect(request.url)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)