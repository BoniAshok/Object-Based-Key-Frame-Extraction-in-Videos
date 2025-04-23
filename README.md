# Object-Based-Key-Frame-Extraction-in-Videos

This is a Flask-powered web application designed for intelligent video analysis. Users can upload a video to automatically generate a summarized version and extract keyframes. The system uses YOLOv5 for object detection, saliency scoring for visual importance, and histogram-based analysis for detecting scene transitions — all seamlessly integrated using OpenCV.

# Key Features
1. Object detection on sampled frames using YOLOv5

2. Extraction of keyframes based on saliency scores

3. Detection of scene transitions using histogram comparisons

4. Automatic summary video creation from keyframes

5. Display of keyframes on the results page for quick viewing

6. Clean, interactive web interface for ease of use

7. Customizable processing options:

      Resize dimensions for frames

      Sampling rate for frame selection

      Batch size for object detection
