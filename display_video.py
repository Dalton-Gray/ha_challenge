from math import floor
from typing import NoReturn

import cv2
import json


def open_video(path: str) -> cv2.VideoCapture:
    """Opens a video file.

    Args:
        path: the location of the video file to be opened

    Returns:
        An opencv video capture file.
    """
    video_capture = cv2.VideoCapture(path)
    if not video_capture.isOpened():
        raise RuntimeError(f'Video at "{path}" cannot be opened.')
    return video_capture


def get_frame_dimensions(video_capture: cv2.VideoCapture) -> tuple[int, int]:
    """Returns the frame dimension of the given video.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        A tuple containing the height and width of the video frames.

    """
    return video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )


def get_frame_display_time(video_capture: cv2.VideoCapture) -> int:
    """Returns the number of milliseconds each frame of a VideoCapture should be displayed.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        The number of milliseconds each frame should be displayed for.
    """
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    return floor(1000 / frames_per_second)


def is_window_open(title: str) -> bool:
    """Checks to see if a window with the specified title is open."""

    # all attempts to get a window property return -1 if the window is closed
    return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1


def open_json_file(path:str) -> dict:
    """Returns the data contained in a JSON file
    
    Args:
        path: an str containing the file path
        
    Returns:
        A dictionary containing the data from the json file specified in the path argument    
    """
    with open(path) as json_file:
        json_data = json.load(json_file)
        json_data = json_data

    return json_data


def main(video_path: str, json_path: str, title: str) -> NoReturn:
    """Displays a video at half size until it is complete or the 'q' key is pressed.

    Args:
        video_path: the location of the video to be displayed
        title: the title to display in the video window
    """

    video_capture = open_video(video_path)
    width, height = get_frame_dimensions(video_capture)
    wait_time = get_frame_display_time(video_capture)
    json_data = open_json_file(json_path)

    try:
        # read the first frame
        success, frame = video_capture.read()

        # create the window
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        # run whilst there are frames and the window is still open
        current_frame = 0 # counter to keep track of which frame we're in
        while success and is_window_open(title):
            current_frame += 1
            # shrink it
            smaller_image = cv2.resize(frame, (floor(width // 2), floor(height // 2)))

            # display it
            cv2.imshow(title, smaller_image)

            # test for quit key
            if cv2.waitKey(wait_time) == ord("q"):
                break

            # read the next frame
            success, frame = video_capture.read()
            
            current_frame_json = json_data.get(str(current_frame))
            # iterates through the list of bounding boxes for the current frame & draw a bounding box for each
            
            detections = [] # change name to pedestrians
            # for bounding_box in current_frame_json.get("bounding boxes"):
            for index, detected_class in enumerate(current_frame_json.get('detected classes')):
                if detected_class == 'person': 
                    x, y, w, h = current_frame_json.get("bounding boxes")[index]
                    detections.append([x, y, w, h])
            
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    VIDEO_PATH = "resources/video_1.mp4"
    JSON_PATH = "resources/video_2_detections.json"
    main(VIDEO_PATH, JSON_PATH, "My Video")
