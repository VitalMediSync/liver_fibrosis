import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os
from glob import glob
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import io
import torch.nn.functional as F
import cv2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Page config and title
st.set_page_config(page_title="Liver Fibrosis Dashboard", layout="wide")

# Sidebar menu
section = st.sidebar.radio("Navigation", [
    "Overview",
    "Data Summary",
    "Training Metrics",
    "Evaluation Report",
    "Inferencing"
])

IMAGE_SIZE = 224 
PRELOADED_IMAGE_PATH = "dataset/"
#PRELOADED_IMAGE_PATH = '/mnt/e/github_source_code/liver_fibrosis/Fibrosis_Dataset/Dataset/'
# Sidebar for model selection
model_selection = st.sidebar.selectbox("Select Model", ["DenseNet121", "Custom"])
model_path = f"models/{model_selection.lower()}/"

# Reload training and evaluation data when model changes
@st.cache_data
def load_training_stats(model_path):
    try:
        return pd.read_json(model_path + "training_metrics.json")
    except:
        return None

@st.cache_data
def load_eval_report(model_path):
    try:
        with open(model_path + "evaluation_metrics.json") as f:
            return json.load(f)
    except:
        return None

training_df = load_training_stats(model_path)
eval_report = load_eval_report(model_path)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ['F0', 'F1', 'F2', 'F3', 'F4']

# --- Model Loading Functions ---

@st.cache_resource
def load_model_densenet():
    """Loads the DenseNet121 model."""
    if os.path.exists(f"{model_path}/liver_fibrosis.pt"):
        model = models.densenet121(weights=None) # Start with no pretrained weights from torchvision
        # Modify the classifier head for 8 classes
        # The original DenseNet121 classifier is nn.Sequential(nn.Dropout(...), nn.Linear(...))
        # We replace the final Linear layer
        num_ftrs = model.classifier.in_features
        #model.classifier = nn.Linear(num_ftrs, len(class_labels)) # Set the output layer to match the number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(0.5), # Assuming dropout was used and saved with the model
            nn.Linear(num_ftrs, len(class_labels)) # Set the output layer to match the number of classes
        )
        model.load_state_dict(torch.load(f"{model_path}/liver_fibrosis.pt", map_location=device))
        model.eval()

        return model
    else:
        st.error(f"DenseNet121 model not found at {model_path}. Inferencing will not work.")
        return None


# Load cached training/evaluation stats
@st.cache_data

def load_training_stats():
    try:
        return pd.read_json(model_path+"training_metrics.json")
    except:
        return None

@st.cache_data

def load_eval_report():
    try:
        with open(model_path+"evaluation_metrics.json") as f:
            return json.load(f)
    except:
        return None

training_df = load_training_stats()
eval_report = load_eval_report()

def preload_images(image_path, image_size=(IMAGE_SIZE, IMAGE_SIZE)):
    # Enhanced grid layout with image selection using button under each image
    st.subheader("Select from preloaded ultrasound images :")
    preloaded_images = []
    preloaded_captions = []

    img_dir = image_path
    # Load dataset details to filter images with Usage == "Testing"
    details_df = pd.read_csv("dataset_details.csv")
    test_images_df = details_df[details_df["Usage"] == "Testing"]

    # Select random 5 samples for each class
    samples_per_class = 1
    sampled_df = (
        test_images_df.groupby("Label", group_keys=False)
        .apply(lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=10))
        .reset_index(drop=True)
    )

    for idx, row in sampled_df.iterrows():
        preloaded_images.append(row["Path"])
        preloaded_captions.append(f"{row['Label']} - {os.path.basename(row['Path'])}")

    selected_image = None
    grid_cols = 5
    grid_rows = len(preloaded_images) // grid_cols
    for row in range(grid_rows):
        cols = st.columns(grid_cols)
        for i in range(grid_cols):
            idx = row * grid_cols + i
            if idx < len(preloaded_images):
                with cols[i]:
                    #st.image(preloaded_images[idx], caption=preloaded_captions[idx], width=120)
                    st.image(img_dir+preloaded_images[idx],  width=120)
                    if st.button(preloaded_captions[idx]):
                        selected_image = img_dir + preloaded_images[idx]
                        st.session_state.selected_image = selected_image

def show_pred_probabilities(probs, pred_idx, prob_df):
    if 'selected_image' in st.session_state:
        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Probability (%)": probs * 100
        })
        colors = ["#1f77b4"] * len(class_labels)
        colors[pred_idx] = "#ff7f0e"
        fig3, ax3 = plt.subplots()
        bars = ax3.bar(prob_df['Class'], prob_df['Probability (%)'], color=colors)
        ax3.set_ylabel("Probability (%)")
        ax3.set_ylim(0, 100)

        # Add probability values on top of each bar
        for bar, prob in zip(bars, prob_df['Probability (%)']):
            height = bar.get_height()
            ax3.annotate(f"{prob:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)
        st.pyplot(fig3)

# Helper function to apply heatmap
def apply_gradcam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    img = np.array(img) / 255.0
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# Updated prediction function with Grad-CAM++
def gradcam_image(orig_img, model, input_tensor, pred_idx, class_names, target_layer):

    st.subheader("Grad-CAM Visualization")
    st.subheader("")

    if model is not None and orig_img is not None:

        heatmap = get_gradcampp_heatmap(model, input_tensor, pred_idx.item(), target_layer)
        cam_img = apply_gradcam_on_image(orig_img, heatmap)

        fig, axes = plt.subplots(figsize=(12, 6))
        axes.imshow(cam_img)
        axes.set_title('')
        axes.axis('off')
        st.pyplot(fig)




def get_gradcampp_heatmap(model, input_tensor, target_class, target_layer):
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]


def load_dataset_details(dataset_details_file="dataset_details.csv"):
    """
    Loads dataset details from a CSV file and displays summary statistics and plots in Streamlit.
    """
    details_df = None
    try:
        details_df = pd.read_csv(dataset_details_file)
        #st.success(f"DataFrame loaded successfully from {dataset_details_file}")
        # Display the first 5 rows with a bigger font using st.markdown and to_html
        st.markdown(
            details_df.head(5).to_html(index=False, escape=False),
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <style>
            table.dataframe {font-size: 18px !important;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # 1. Distribution of images by Label
        st.subheader("Distribution of Images by Fibrosis Level")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Label', data=details_df, ax=ax1, order=sorted(details_df['Label'].unique()), width=0.4, palette='viridis')
        ax1.set_title('Distribution of Images by Fibrosis Level')
        ax1.set_xlabel('Fibrosis Level')
        ax1.set_ylabel('Number of Images')
        st.pyplot(fig1)
        plt.close(fig1) # Close the figure to free up memory

        # 2. Distribution of images by Usage (Train, Validation, Test)
        st.subheader("Distribution of Images by Dataset Split")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Usage', data=details_df, ax=ax2, order=['Training', 'Validation', 'Testing'],  width=0.3, palette='viridis') # Corrected column name
        ax2.set_title('Distribution of Images by Dataset Split')
        ax2.set_xlabel('Dataset Split')
        ax2.set_ylabel('Number of Images')
        st.pyplot(fig2)
        plt.close(fig2)

        # 3. Distribution of image sizes (Kernel Density Estimate plot)
        st.subheader("Distribution of Image Sizes (KB)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(details_df['Size (KB)'], bins=50, kde=True, ax=ax3)
        ax3.set_title('Distribution of Image Sizes (KB)')
        ax3.set_xlabel('Size (KB)')
        ax3.set_ylabel('Frequency')
        st.pyplot(fig3)
        plt.close(fig3)

        # 4. Distribution of images by Grayscale/RGB
        st.subheader("Distribution of Images by Color Mode")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Color Type', data=details_df, ax=ax4, width=0.3, palette='viridis') # Corrected column name
        ax4.set_title('Distribution of Images by Color Mode')
        ax4.set_xlabel('Color Mode')
        ax4.set_ylabel('Number of Images')
        st.pyplot(fig4)
        plt.close(fig4)

        # 5. Distribution of images by Resolution (Top N resolutions)
        st.subheader("Top Image Resolutions")
        # Count occurrences of each resolution
        resolution_counts = details_df['Resolution'].value_counts().reset_index()
        resolution_counts.columns = ['Resolution', 'Count']

        # Display top 10 resolutions (adjust as needed)
        top_n_resolutions = 10
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Resolution', y='Count', data=resolution_counts.head(top_n_resolutions), ax=ax5, width=0.3, palette='viridis')
        ax5.set_title(f'Top {top_n_resolutions} Image Resolutions')
        ax5.set_xlabel('Resolution')
        ax5.set_ylabel('Number of Images')
        ax5.tick_params(axis='x', rotation=45) # Rotate labels for better readability
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        st.pyplot(fig5)
        plt.close(fig5)

        # 6. Count of images by Label and Usage
        st.subheader('Distribution of Images by Fibrosis Level and Dataset Split')
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        sns.countplot(x='Label', hue='Usage', data=details_df, order=sorted(details_df['Label'].unique()), hue_order=['Training', 'Validation', 'Testing'], ax=ax6)
        ax6.set_title('Distribution of Images by Fibrosis Level and Dataset Split')
        ax6.set_xlabel('Fibrosis Level')
        ax6.set_ylabel('Number of Images')
        st.pyplot(fig6)
        plt.close(fig6)

    except FileNotFoundError:
        st.error(f"Error: The file '{dataset_details_file}' was not found.")
    except Exception as e:
        st.error(f"An error occurred while processing the dataset details: {e}")


# Show classification report as a bar graph for each class

def show_classification_report(eval_report):
    if eval_report is not None:
        if "classification_report" in eval_report:
            # Show classification report as a bar graph for each class
            if "classification_report" in eval_report:
                class_report = eval_report["classification_report"]
                # Convert to DataFrame
                class_names = list(class_report.keys())
                metrics = ["precision", "recall", "f1-score"]
                valid_class_names = [cls for cls in class_names if isinstance(class_report[cls], dict)]
                data = {metric: [class_report[cls][metric] for cls in valid_class_names] for metric in metrics}
                df = pd.DataFrame(data, index=valid_class_names)
                fig, ax = plt.subplots(figsize=(8, 5))
                df.plot(kind="bar", ax=ax)
                ax.set_title("Classification Metrics per Class")
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1)
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.info("Classification report not available.")

# Show confusion matrix
def show_confusion_matrix(eval_report):
    if eval_report is not None:
        if "confusion_matrix" in eval_report:
            cm = np.array(eval_report['confusion_matrix'])
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=eval_report['class_names'], yticklabels=eval_report['class_names'])
            st.pyplot(fig2)


def show_roc_curve(eval_report):
    if eval_report is not None: 
        if "roc_curve" in eval_report:
            # Show ROC curve if data is available
            if eval_report["roc_curve"]:
                fpr, tpr, roc_auc = {}, {}, {}

                for i in eval_report["roc_curve"]["auc"].keys():
                    fpr[i], tpr[i] = eval_report["roc_curve"]['fpr'][i], eval_report["roc_curve"]['tpr'][i]
                    roc_auc[i] = auc(fpr[i], tpr[i])

                fig3, ax3 = plt.subplots()
                for i in eval_report["roc_curve"]["auc"].keys():
                    ax3.plot(fpr[i], tpr[i], label=f"{class_labels[int(i)]} (AUC = {roc_auc[i]:.2f})")
                ax3.set_xlabel("False Positive Rate")
                ax3.set_ylabel("True Positive Rate")
                ax3.set_title("Receiver Operating Characteristic (ROC) Curve")
                ax3.legend(loc="lower right")
                st.pyplot(fig3)
            else:
                st.info("ROC curve data not available.")


def show_project_overview():
    st.title("Liver Fibrosis Classification")
    st.markdown("""
    This project focuses on the classification of Liver Fibrosis, a critical aspect of assessing liver health. Utilizing ultrasound as a non-invasive imaging method and deep learning techniques, the aim is to categorize fibrosis into various stages (e.g., F0-F4). This approach offers a less intrusive alternative to traditional biopsy. The initiative provides a clear overview of the clinical problem and the role of advanced computational techniques in supporting medical diagnosis through image analysis. The ultimate goal is to facilitate early and accurate assessment of liver fibrosis for improved patient management.

    **Features:**
    - Data distribution and sample images
    - Training and validation performance tracking
    - Evaluation metrics and confusion matrix
    - Upload and test your own ultrasound image
    """)
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        # Placeholder for an image
        st.image("logo.png", width=700, caption="Liver Fibrosis")


TOTAL_SLIDES = 5
def slide_show():
    images = [] 
    for i in range (TOTAL_SLIDES):
        images.append(f"slides/{i+1}.png")

    # Initialize session state for slide index
    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0

    col01, col02 = st.columns([1, 0.28])
    with col01:
        # Show the current image
        st.image(images[st.session_state.slide_index], use_container_width=True)

    col1, col2 = st.columns([1, 1])
    # Button controls
    with col1:
        if st.button("< Back"):
            if st.session_state.slide_index > 0:
                st.session_state.slide_index -= 1
            else:
                st.session_state.slide_index = len(images) - 1  # Loop to last image
    with col2:
        if st.button("Next >"):
            if st.session_state.slide_index < len(images) - 1:
                st.session_state.slide_index += 1
            else:
                st.session_state.slide_index = 0  # Loop to first image

    # Optional: Show slide number
    #st.write(f"Image {st.session_state.slide_index + 1} of {len(images)}")

# Section: Overview
if section == "Overview":
    #show_project_overview()
    slide_show()

# Section: Data Summary
elif section == "Data Summary":
    st.title("Data Summary")
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        st.subheader("Ultrasound Image Overview")
        load_dataset_details()


# Section: Training Metrics
elif section == "Training Metrics":
    st.title("Training Progress")
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        if training_df is not None:
            st.subheader("Loss over Epochs")
            st.line_chart(training_df[['train_losses', 'val_losses']])

            st.subheader("Accuracy over Epochs")
            st.line_chart(training_df[['train_accuracies', 'val_accuracies']])
        else:
            st.warning("Training metrics not found.")


# Section: Evaluation Report
elif section == "Evaluation Report":
    st.title("Model Evaluation")
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        if eval_report is not None:
            st.subheader("Overall Test Accuracy")
            st.markdown(f"### **{eval_report['classification_report']['accuracy']*100:.2f}%**")

            st.subheader("Classification Report")
            show_classification_report(eval_report)

            st.subheader("Confusion Matrix")
            show_confusion_matrix(eval_report)

            st.subheader("ROC Curve")
            show_roc_curve(eval_report)

        else:
            st.warning("Evaluation report not available.")


# Section: Inference
elif section == "Inferencing":
    st.title("Inferencing")

    model = None
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transform(image.convert('RGB')).unsqueeze(0)

    if model_selection == "DenseNet121":
        model = load_model_densenet()
        

    selected_image = None
    preload_images(PRELOADED_IMAGE_PATH)

    col1, col2, col3 = st.columns([0.8, 1, 1.2])
    image = None
    probs = None
    pred_idx = None
    pred_label = None

    with col1:
        st.subheader("Upload ultrasound image :")

        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width =True)
        elif "selected_image" in st.session_state:
            selected_image = st.session_state.selected_image
            image = Image.open(selected_image)
            st.image(image, caption=f"Selected Image ({selected_image})", use_container_width =True)
        else:
            st.info("Please select or upload an image to begin inference.")

        # Load the model and do prediction
        if model is not None and image is not None:
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0).numpy()
                pred_idx = np.argmax(probs)
                pred_label = class_labels[pred_idx]
    with col2:
        # Grad-CAM visualization ---
        if model is not None and image is not None:
            inference_transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            target_layer = model.features.norm5
            gradcam_image(image, model, input_tensor, pred_idx, class_labels, target_layer)  

    with col3:
        if model is not None and image is not None:
            st.markdown(f"### Predicted Class: **{pred_label}**")
            st.subheader("")

            prob_df = pd.DataFrame({
                    'Class': class_labels,
                    'Probability (%)': probs * 100
            }).sort_values(by='Probability (%)', ascending=False) # Sort for better readability
            show_pred_probabilities(probs, pred_idx, prob_df)




