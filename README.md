# Deep-Learning-Enhanced-Diagnostic-Systems-Integrating-Convolutional-Neural-Networks-and-Grad-CAM
This tool is a deep learning-powered web application for diagnosing Monkeypox lesions in skin images. Built with a pre-trained ResNet50 model and Grad-CAM visualizations, the application accurately classifies skin images as Monkeypox or non-Monkeypox, providing confidence scores and heatmaps to highlight areas of focus.
Key Features
Image Classification: Uses ResNet50, fine-tuned to classify Monkeypox vs. non-Monkeypox cases.
Grad-CAM Explanations: Generates heatmaps to show areas that influenced the model’s decision, helping improve interpretability.
Real-Time Alerts: Sends email alerts with diagnostic reports if Monkeypox is detected.
Automated PDF Reports: Provides a downloadable PDF containing predictions, confidence levels, and follow-up recommendations.
Flask Web Interface: Deploys as a user-friendly web app, allowing users to upload images and receive real-time results.
