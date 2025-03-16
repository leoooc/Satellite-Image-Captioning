# Satellite-Image-Captioning
Satellite Image Captioning with ViT + Decoder
This project implements a pipeline to generate descriptive captions for satellite images using a combination of a Vision Transformer (ViT) and an LSTM-based decoder. The approach leverages state-of-the-art vision models for robust image feature extraction and couples it with a sequential language model for caption generation.

# Overview
Objective:
Generate natural language descriptions for satellite images. The model uses a pretrained ViT as a fixed feature extractor and a bidirectional LSTM as the decoder to generate captions word-by-word.


Project Structure
Data Handling:
The code reads the CSV files, flattens multiple captions (if any) per image into separate samples, cleans the text (lowercasing and removing punctuation), and builds a vocabulary. Images are preprocessed (resized and normalized) to meet the input requirements of the ViT encoder.

Model Architecture:

Encoder: Uses a pretrained ViT (ViT-B/16) to extract image features. The ViT's parameters are frozen, and its output is projected to match the decoder’s embedding size.
Decoder: A bidirectional LSTM generates captions based on the image embedding and the sequence of word embeddings. Special tokens (e.g., <pad>, <unk>, <end>) are used to manage the sequence flow.
Training Pipeline:
The training process employs teacher forcing where the correct caption tokens are fed into the decoder during training. Cross-entropy loss (ignoring padding tokens) is used to update the decoder’s weights via the Adam optimizer.

Caption Generation:
For inference, the image is encoded and then the decoder generates a caption sequentially, word-by-word, until an <end> token is produced or a maximum length is reached. The output is visualized by displaying the test images alongside their generated captions.

Dependencies
Python 3.7+
PyTorch and torchvision
Pandas
NumPy
Pillow (PIL)
Matplotlib
Key Considerations
Path Management:
The CSV files contain paths like train/airport_101.jpg. The code has been adjusted to avoid duplicate folder names by using the parent dataset directory when constructing the full file path.

# Final Thoughts
This project serves as a forward-thinking baseline for satellite image captioning. By integrating powerful vision models with sequential language generation, it aims to generate meaningful and contextually rich descriptions of satellite imagery. The current implementation is modular and flexible, offering plenty of avenues for future improvement and adaptation.
