# CapraCare

CapraCare is a deep-learning system for automated goat skin disease classification, built as part of an undergraduate research project at Thapar Institute of Engineering and Technology.

## Overview
The project curates a self-collected, manually labelled image dataset of goat skin conditions and applies transfer learning and data augmentation to classify diseases with high accuracy despite limited sample size.

## Highlights
- Custom CNN (DLAB-Net v2) and fine-tuned EfficientNet-B0 baseline
- 97.77% classification accuracy on validation
- Grad-CAM visualizations for model interpretability
- ONNX export for lightweight deployment
- Web and mobile app for easy accessibility

## Tech Stack
Python, TensorFlow/Keras, OpenCV, EfficientNet-B0, ONNX

## Repository Structure
- 	rain_classifier.py — model training pipeline
- ugment_dataset.py, ebuild_dataset.py, estratify_classification.py — dataset preparation
- gradcam_visualizer.py — Grad-CAM interpretability
- convert_to_onnx.py — ONNX export
- pp.py — application entry point
- plot_training_curves.py, plot_confusion_matrices.py — evaluation and reporting

## Status
Manuscript in preparation.
