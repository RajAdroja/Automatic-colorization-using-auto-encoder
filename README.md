# Colorization Autoencoder

This project implements a convolutional autoencoder for image colorization using the CIFAR-10 dataset. The autoencoder is trained to convert grayscale images into their colorized versions.

## Features

- Converts RGB images to grayscale as input.
- Trains an encoder-decoder convolutional neural network to colorize grayscale images.
- Includes visualization of original and colorized images for comparison.
- Saves the trained model in HDF5 format for reuse.

## Technologies Used

- **Python**: Core programming language.
- **Keras**: For building the encoder-decoder model.
- **NumPy**: For efficient numerical computations.
- **OpenCV**: For image processing and grayscale conversion.
- **Matplotlib**: For visualizing the results.
- **Google Colab**: For running the training and inference scripts.

## Dataset

The CIFAR-10 dataset is used for training and testing. It contains 60,000 32x32 color images in 10 classes, with 50,000 training and 10,000 testing images.

## Project Structure

- **Encoder Model**: Compresses the grayscale input image to a latent vector.
- **Decoder Model**: Reconstructs the colorized image from the latent vector.
- **Autoencoder Model**: Combines the encoder and decoder for end-to-end training.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/colorization-autoencoder.git
   cd colorization-autoencoder
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib opencv-python tensorflow keras
   ```

3. Download the CIFAR-10 dataset (handled automatically by the code).

## Usage

### 1. Import Libraries and Load Data
The project uses TensorFlow/Keras and OpenCV for preprocessing and model building.

### 2. Preprocess Images
- Convert RGB images to grayscale using OpenCV.
- Normalize and reshape the images for the model.

### 3. Train the Autoencoder
Run the training script:
```python
autoencoder.fit(x_train_Gray, x_train, validation_data=(x_test_Gray, x_test), epochs=30, batch_size=32, callbacks=callbacks)
```

### 4. Save and Evaluate the Model
- Save the trained model:
  ```python
  autoencoder.save('colourization_model.h5')
  ```
- Use the trained model to predict colorized images.

### 5. Visualize Results
Visualize the original and colorized images side by side using Matplotlib.

### Results
Example output:
- **Original Images**: Grayscale images from the CIFAR-10 dataset.
- **Colorized Images**: Predicted colorized versions of the input grayscale images.

## Example Commands
To train the model and visualize results:
```bash
python train_autoencoder.py
python visualize_results.py
```

## Model Architecture

### Encoder
- Conv2D layers to extract features.
- Latent vector for dimensionality reduction.

### Decoder
- Conv2DTranspose layers to reconstruct the image from the latent vector.

### Autoencoder
- Combines the encoder and decoder models.

## Callbacks
- **ReduceLROnPlateau**: Adjusts the learning rate based on validation loss.
- **ModelCheckpoint**: Saves the best model during training.

## Visualizations
- **Original Grayscale Images**
- **Colorized Images** (Predicted)

## Save and Download Model
The trained model is saved as an HDF5 file and can be downloaded for later use.

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Keras Documentation](https://keras.io)

---

Feel free to modify this README file as needed!
