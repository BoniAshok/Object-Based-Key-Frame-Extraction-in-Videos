# Object-Based-Key-Frame-Extraction-in-Videos

This is a Flask-powered web application designed for intelligent video analysis. Users can upload a video to automatically generate a summarized version and extract keyframes. The system uses YOLOv5 for object detection, saliency scoring for visual importance, and histogram-based analysis for detecting scene transitions — all seamlessly integrated using OpenCV.

#Key Features
Object detection on sampled frames using YOLOv5

Extraction of keyframes based on saliency scores

Detection of scene transitions using histogram comparisons

Automatic summary video creation from keyframes

Display of keyframes on the results page for quick viewing

Clean, interactive web interface for ease of use

Customizable processing options:

Resize dimensions for frames

Sampling rate for frame selection

Batch size for object detection
