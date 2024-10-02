# IPDC Classification Model Training

# Thema IPDC Model Training

This Python script trains a multi-label text classification model using the Hugging Face Transformers library. The model is designed to classify text descriptions into multiple thematic categories based on a CSV dataset.

### Required Packages

You can install the necessary Python packages with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Dataset

The dataset (`producten_en_diensten_2024-09-13_21-47-37.csv`) should be located in your home directory (`~`). This CSV file contains textual descriptions and corresponding thematic labels. The script reads this data, processes it, and uses it to train a text classification model.

Ensure your CSV file contains the following fields:
- `thema`: The thematic labels for each description.
- `tpe`: Type of the entry.
- `beschrijving`: The textual description.

### 2. Model

The script uses the following pre-trained Hugging Face model:

- **Hugging Face Model:** `papluca/xlm-roberta-base-language-detection`

### 3. Configuration

- `DATAFILE`: The name of your CSV file (`producten_en_diensten_2024-09-13_21-47-37.csv`).
- `HUGGINGFACE_MODEL`: The pre-trained model to fine-tune (default: `"papluca/xlm-roberta-base-language-detection"`).
- `ML_FLOW_URI`: The URI for your MLflow server, if needed (default: `"http://localhost:5000"`).

### 4. Running the Script

To train the model, simply execute the script:

```bash
python train_model.py
```

The script will:
1. Load the dataset.
2. Tokenize and preprocess the data.
3. Train a multi-label classifier.
4. Evaluate the model performance using accuracy, F1 score, precision, and recall.
5. Save the trained model and logs.

### 5. Metrics and Evaluation

The model uses a sigmoid activation function for predictions and computes the following metrics during evaluation:
- Accuracy
- F1 Score
- Precision
- Recall

### 6. Logging

The script includes logging for errors, and the model training time is printed at the end of execution.

If you want to track the training process using MLflow, uncomment the following lines:

```python
# mlflow.set_tracking_uri(ML_FLOW_URI)
# mlflow.set_experiment('thema-ipdc-model')
```

Ensure your MLflow server is running and accessible.

## Example Output

```bash
Done. Model training ran for  1234.56 seconds.
```
