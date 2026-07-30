# Dog vs Wolf AI Classifier - Group CO6

## Overview
This repository contains a deep learning computer vision web application designed to accurately classify images of dogs and wolves. Developed as a group project for the engineering course GET324, this project showcases the practical application of cloud computing and AI model deployment. We successfully designed, trained, evaluated, and deployed a Convolutional Neural Network (CNN) capable of performing binary image classification, utilizing industry-standard tools like Streamlit, GitHub, and Kaggle.

---

## Demo
Upload any image of a dog or a wolf, and the model returns the predicted class along with real-time confidence scores for each category. 
**Live Application:** [Group CO6 Classifier App](https://dog-vs-wolves-ai-classifier-group-co6.streamlit.app/)

##

## Dataset
Dogs vs Wolves Dataset — Kaggle 
**Dataset Link :** [Dogs vs Wolves Dataset](https://www.kaggle.com/datasets/harishvutukuri/dogs-vs-wolves)


---

## Tech Stack
*   **Python 3.11+**
*   **Deep Learning Framework:** TensorFlow / Keras (v2.16.1)
*   **Frontend UI:** Streamlit
*   **Data Processing:** NumPy, Pillow (PIL)

---
## Contributors
*  **Ohotuowo Morgan Agrinya	-	@ohotuowo-morgan**
*  **Usanga Ofonmbuk Mfon	-	@ofon4real1**
*  **Ekong Ime Ime	-	@imeekong**

---

##  Repository Structure
```text
group-co6-vision-ai-model-/
├── .devcontainer/              # Configurations for consistent development environments
├── .gitignore                  # Excludes unnecessary local files from version control
├── README.md                   # Detailed project documentation (this file)
├── app.py                      # Main Streamlit frontend application and logic
├── dogs_v_wolves.ipynb         # Jupyter Notebook containing the training and evaluation pipeline
├── mobilenetv3_transfer.keras  # The compiled and trained Keras 3 CNN model archive
└── requirements.txt            # Python dependencies for cloud deployment




