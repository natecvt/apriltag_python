import cv2
import numpy as np
from os import PathLike
import yaml
from dt_apriltags import Detector

def load_config(path: str | PathLike):

    if not path.endswith((".yaml", ".yml")):
        print("Path does not specify a config")
        return None

    with open(path, "r") as stream:
        params = yaml.safe_load(stream)

        if params is None:
            print("Invalid file syntax or data")
            return None

        return params

def init_capture_apriltags(confPath: str | PathLike) -> list[cv2.VideoCapture, Detector]:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        exit(1)

    params = load_config(confPath)

    param_list = [params["family"], 
                  params["nthreads"], 
                  params["quad_decimate"], 
                  params["quad_sigma"], 
                  params["refine_edges"], 
                  params["decode_sharpening"],
                  params["debug"]]

    pcheck = param_list.count(None) == 0

    if not pcheck:
        print("Apriltag detector not initialized")
        exit(2)
    
    detector = Detector(families=param_list[0],
                        nthreads=param_list[1],
                        quad_decimate=param_list[2],
                        quad_sigma=param_list[3],
                        refine_edges=param_list[4],
                        decode_sharpening=param_list[5],
                        debug=param_list[6])
    
    if detector is None:
        exit(3)

    print("Apriltag detector initialized")

    return [cap, detector]

def capture_image(cap: cv2.VideoCapture) -> cv2.typing.MatLike | None:
    ret, frame = cap.read()
    if not ret:
        print("Image failed to capture")
        return None
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return image
    
def detect_apriltags(image: cv2.typing.MatLike, detector: Detector):
    detector.detect(image)

def main():
    init_capture_apriltags("config.yaml")

if __name__ == "__main__":
    main()

