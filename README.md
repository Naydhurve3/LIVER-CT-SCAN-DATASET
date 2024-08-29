# Liver Cancer Detection Using CNN
## Overview
This project focuses on the detection and classification of liver cancer using Convolutional Neural Networks (CNNs). By leveraging a comprehensive dataset of liver CT scan images, the project aims to accurately identify cancerous regions within the liver. The model is trained and evaluated on real-world medical imaging data to ensure robust performance in clinical scenarios.

## Project Structure
Data Preprocessing: Techniques such as normalization, augmentation, and resizing are applied to enhance the quality of the CT scan images and prepare them for model training.

Model Development: Various CNN architectures are explored and tested to find the most suitable model for liver cancer detection. This includes fine-tuning hyperparameters, optimizing the model structure, and implementing dropout and regularization techniques to prevent overfitting.

Model Training: The models are trained on a carefully curated dataset, utilizing GPU acceleration to handle the computational load. The training process includes monitoring metrics such as accuracy, precision, recall, and F1-score to ensure the model is learning effectively.

Model Evaluation: Rigorous testing is conducted using a separate test dataset to evaluate the performance of the trained models. Visualization tools are used to display the results, including confusion matrices, ROC curves, and other relevant performance indicators.

Deployment: The final model is deployed for real-world use in medical imaging applications, where it can assist radiologists and medical professionals in detecting liver cancer with higher accuracy and efficiency.

## Dataset
Source: The dataset consists of liver CT scan images labeled for cancer detection.

Dataset : `https://www.kaggle.com/datasets/andrewmvd/lits-png`

Structure: The dataset includes the following columns in a CSV file:

filepath: Path to the CT scan image.
liver_maskpath: Path to the liver mask image.
tumor_maskpath: Path to the tumor mask image.
liver_mask_empty: Boolean indicating whether the liver mask is empty.
tumor_mask_empty: Boolean indicating whether the tumor mask is empty.
Images: There are 58,638 CT scan images in total, divided into training and testing sets with a 70:30 ratio.

## Model Training Details
Data Split: Only 13% of the total training data and 23% of the total testing data are used to minimize training time.
Epochs: The model is trained for 10 epochs.
Validation: TFRecord files are used for train, test, and validation datasets.
Evaluation Metrics: The model's performance is evaluated using accuracy, precision, recall, and F1-score.
How to Run the Project
Clone the Repository:

```bash
Copy code

git clone https://github.com/Naydhurve3/LIVER-CT-SCAN-DATASET.git

cd liver-cancer-detection
```
Install Dependencies: Ensure that you have Python installed.
 ## Install the required libraries:

```bash
Copy code
pip install -r requirements.txt
```
Prepare the Dataset: Place the dataset in the appropriate directory as specified in the project code.

Train the Model: Run the training script to train the model on the dataset:

bash
Copy code
python train_model.py
Evaluate the Model: Evaluate the trained model using the test dataset:

```bash
Copy code

python evaluate_model.py
```
Visualize Results: View the performance metrics and visualizations:

```bash
Copy code

python visualize_results.py
```
## Results
The project demonstrates promising results in the detection and classification of liver cancer from CT scan images. The trained model achieves high accuracy and robustness, making it a valuable tool for medical imaging applications.

## Future Work
Data Augmentation: Explore advanced data augmentation techniques to further improve model performance.
Transfer Learning: Implement transfer learning from pre-trained models to enhance the model's ability to generalize.
Real-Time Detection: Develop a real-time detection system that can process and analyze CT scans on the fly.
Contributing
Contributions are welcome! If you have any ideas, improvements, or bug fixes, feel free to submit a pull request.

## Contact
For any inquiries or questions, feel free to reach out:

Email: nayandhurve44@gmail.com

LinkedIn: www.linkedin.com/in/nayan-dhurve-31815a258
