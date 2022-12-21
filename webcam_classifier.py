# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run video classification."""

import argparse
import sys
import time

import numpy as np 

import cv2

from tensorflow import keras
import tensorflow as tf

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_MODEL_FPS = 7  # Ensure the input images are fed to the model at this fps.
_MODEL_FPS_ERROR_RANGE = 0.1  # Acceptable error range in fps.
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def get_sequence_model():
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    optimizer = tf.keras.optimizers.legacy.Adam()

    rnn_model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return rnn_model

def prepare_single_video(frames, feature_extractor):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(sequence_model, frames , feature_extractor):
    class_vocab = ['landslide', 'no_landslide']

    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    print(probabilities[0])
    # if probabilities[0] < 0.5:
    #   print(f" {class_vocab[0]}: {(1-probabilities[0]) * 100:5.2f}%")
    # else:
    #   print(f"  {class_vocab[1]}: {probabilities[0] * 100:5.2f}%")
    return probabilities[0], frames

def run(model: str, label: str, max_results: int, num_threads: int,
        camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the TFLite video classification model.
      label: Name of the video classification label.
      max_results: Max of classification results.
      num_threads: Number of CPU threads to run the model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """
  # Initialize the video classification model
#   options = VideoClassifierOptions(
#       num_threads=num_threads, max_results=max_results)
#   classifier = VideoClassifier(model, label, options)

  feature_extractor = build_feature_extractor()

  sequence_model = get_sequence_model()
  sequence_model.load_weights("video_classifier/")

  # Variables to calculate FPS
  counter, fps, last_inference_start_time, time_per_infer = 0, 0, 0, 0

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  vid_frame_curr = []

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    counter += 1      

    # Mirror the image
    image = cv2.flip(image, 1)

    # Ensure that frames are feed to the model at {_MODEL_FPS} frames per second
    # as required in the model specs.
    current_frame_start_time = time.time()
    diff = current_frame_start_time - last_inference_start_time
    if diff * _MODEL_FPS >= (1 - _MODEL_FPS_ERROR_RANGE):
      # Store the time when inference starts.
      last_inference_start_time = current_frame_start_time

      # Calculate the inference FPS
      fps = 1.0 / diff

      # Convert the frame to RGB as required by the TFLite model.
      frame = crop_center_square(image)
      frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
      frame_rgb = frame[:, :, [2, 1, 0]]
    #   frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      if counter < _MODEL_FPS:
        vid_frame_curr.append(frame_rgb)  

      else:
        # Feed the frame to the video classification model.
        probability, frame = sequence_prediction(sequence_model,np.array(vid_frame_curr), feature_extractor)

        vid_frame_curr = vid_frame_curr[1:]
        vid_frame_curr.append(frame_rgb)

    #   categories = classifier.classify(frame_rgb)

    # Calculate time required per inference.
    time_per_infer = time.time() - current_frame_start_time

    # Notes: Frames that aren't fed to the model are still displayed to make the
    # video look smooth. We'll show classification results from the latest
    # classification run on the screen.
    # Show the FPS .
    fps_text = 'Current FPS = {0:.1f}. Expect: {1}'.format(fps, _MODEL_FPS)
    text_location = (_LEFT_MARGIN, _ROW_SIZE)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Show the time per inference.
    time_per_infer_text = 'Time per inference: {0}ms'.format(
        int(time_per_infer * 1000))
    text_location = (_LEFT_MARGIN, _ROW_SIZE * 2)
    cv2.putText(image, time_per_infer_text, text_location,
                cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR,
                _FONT_THICKNESS)

    # Show classification results on the image.
    if counter >_MODEL_FPS:
        if probability < 0.5:
            class_name = "landslide"
        else:
            class_name = "no_landslide"
        probability = round(probability, 2)
        result_text = class_name + ' (' + str(probability) + ')'
        # Skip the first 2 lines occupied by the fps and time per inference.
        text_location = (_LEFT_MARGIN, (0 + 3) * _ROW_SIZE)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('video_classification', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of video classification model.',
      required=False,
      default='movinet_a0_int8.tflite')
      # default='movienet_a0_base.tflite')
  parser.add_argument(
      '--label',
      help='Name of video classification label.',
      required=False,
      default='kinetics600_label_map.txt')
  parser.add_argument(
      '--maxResults',
      help='Max of classification results.',
      required=False,
      default=2)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, args.label, int(args.maxResults), int(args.numThreads),
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()