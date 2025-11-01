import sys
import onnxruntime as ort
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Processing')
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        self.left_label = QLabel(self)
        self.left_label.setAlignment(Qt.AlignCenter)
        self.right_label = QLabel(self)
        self.right_label.setAlignment(Qt.AlignCenter)

        image_layout.addWidget(self.left_label)
        image_layout.addWidget(self.right_label)
        
        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.clicked.connect(self.upload_image)
        
        self.process_button = QPushButton('Process Image', self)
        self.process_button.clicked.connect(self.process_image)

        layout.addWidget(self.upload_button)
        layout.addWidget(self.process_button)
        layout.addLayout(image_layout)
        
        self.setLayout(layout)

        self.model_path = "model.onnx"
        self.ort_session = self.load_model(self.model_path)

        self.original_image = None

    def load_model(self, model_path):
        return ort.InferenceSession(model_path)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.bmp)')
        
        if file_name:
            self.original_image = Image.open(file_name)
            self.display_image(self.original_image, self.left_label)

    def display_image(self, image, label):
        image = image.convert('RGB')
        img_array = np.array(image)
        h, w, c = img_array.shape
        bytes_per_line = 3 * w
        q_image = QPixmap.fromImage(img_array.data)
        label.setPixmap(q_image.scaled(400, 400, Qt.KeepAspectRatio))

    def process_image(self):
        if self.original_image:
            processed_image = self.run_inference(self.original_image)
            self.display_image(processed_image, self.right_label)

    def run_inference(self, image):
        img_array = np.array(image.convert('RGB'))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        input_size = (640, 640)
        img_resized = cv2.resize(img_array, input_size)

        img_normalized = img_resized / 255.0
        img_normalized = np.transpose(img_normalized, (2, 0, 1))
        img_normalized = np.expand_dims(img_normalized, axis=0).astype(np.float32)

        inputs = {self.ort_session.get_inputs()[0].name: img_normalized}
        outputs = self.ort_session.run(None, inputs)
        
        processed_image = img_resized
        processed_image = Image.fromarray(processed_image)
        return processed_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())
