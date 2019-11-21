from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import numpy as np
import datetime
import argparse
from keras.models import load_model
import os
import json
import random
from collections import Counter
from google.cloud import texttospeech
from playsound import playsound
import time
import winsound


# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized


# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.types.VoiceSelectionParams(
    language_code='en-US',
    ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

# Select the type of audio file you want returned
audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.MP3)

detection_graph, sess = detector_utils.load_inference_graph()
labels = json.load(open("../signclassification/labels.json"))
model = load_model("../signclassification/model.h5.olu.old")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        all_images = detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                                      scores, boxes, im_width, im_height,
                                                      image_np)

        majority_list = []
        if len(all_images) != 0:

            # print(np.shape(np.array(all_images)))
            # dell = cv2.imread("../signclassification/data/asl_alphabet_train/asl_alphabet_train/A/A3_custom_9207.jpg")
            # dell = cv2.resize(dell, (64, 64))
            # cv2.imshow("title2", dell)
            # prediction = model.predict(np.array([dell]))
            # prediction_indices = tf.keras.backend.eval(tf.argmax(prediction, axis=1))
            # print(f"This is the fake prediction: {prediction_indices}: {[labels[str(idx)] for idx in prediction_indices]}")

            prediction = model.predict(np.array(all_images))
            cv2.imshow("title", all_images[0])
            prediction_indices = tf.keras.backend.eval(tf.argmax(prediction, axis=1))
            print(f"This is the honest prediction: {prediction_indices}: {[labels[str(idx)] for idx in prediction_indices]}")
            majority_list.extend([labels[str(idx)] for idx in prediction_indices])

        # # Calculate Frames per second (FPS)

        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        # if num_frames % 2 == 0:
        #     for img in all_images:
        #         print("IMAGE HERE!!!")
        #         cv2.imwrite(f"../signclassification/data2/A/A{num_frames//10}_custom_{random.randint(0,10000)}.jpg", img)

        if num_frames % 10 == 0 and len(majority_list) > 0:
            cnt = Counter(majority_list)
            majority_list = []
            majority = cnt.most_common(1)[0]
            print(f"IMAGE HERE!!!: {majority}")
            synthesis_input = texttospeech.types.SynthesisInput(text=majority[0])
            response = client.synthesize_speech(synthesis_input, voice, audio_config)
            appendix = random.randint(0, 10000)
            # winsound.PlaySound(response.audio_content)
            # time.sleep(1000)
            with open(f'output-{appendix}.mp3', 'wb') as out:
                # Write the response to the output file.
                out.write(response.audio_content)
                print('Audio content written to file "output.mp3"')
                out.close()
            playsound(f'output-{appendix}.mp3', block=True)

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(num_frames/10)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(27) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
