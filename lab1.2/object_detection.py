import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# import utils
import mediapipe as mp
import numpy as np
import matplotlib.pylab as plt
from picamera2 import Picamera2

MODEL = "efficientdet_lite0.tflite"

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


class ObjectDetector:
    def __init__(self):
        self.picam2 = Picamera2()
        # Optimize for still photos
        config = self.picam2.create_still_configuration()
        self.picam2.configure(config)
        self.picam2.start()

        # Initialize the object detection model
        base_options = python.BaseOptions(model_asset_path=MODEL)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, score_threshold=0.3, max_results=3
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

        self.count = 0

    def visualize(
        self,
        image: np.ndarray,
        detection_result: vision.ObjectDetectorResult,
    ) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.

        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to be visualize.

        Returns:
            Image with bounding boxes.
        """
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + " (" + str(probability) + ")"
            print(result_text)
            text_location = (
                _MARGIN + bbox.origin_x,
                _MARGIN + _ROW_SIZE + bbox.origin_y,
            )
            cv2.putText(
                image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE,
                _TEXT_COLOR,
                _FONT_THICKNESS,
            )

        return image

    def detect(self) -> vision.ObjectDetectorResult:
        image = self.picam2.capture_array()
        # image = rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = image

        # Create a TensorImage object from the RGB image.
        input_tensor = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection estimation using the model.
        detection_result = self.detector.detect(input_tensor)

        # Draw keypoints and edges on input image
        image = self.visualize(rgb_image, detection_result)

        plt.imshow(image)
        plt.axis("off")  # Hide axes
        plt.savefig(
            "captured_images/" + str(self.count) + "file_name",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        self.count += 1

        return detection_result
