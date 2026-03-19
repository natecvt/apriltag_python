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

def init_capture_apriltags(params) -> list[cv2.VideoCapture, Detector]:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not found")
        exit(1)

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
        print("Detector failed to initialize")
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
    
def detect_apriltags(image: cv2.typing.MatLike, detector: Detector, params) -> list | None:
    cam_vals = (params["intrinsics"]["fx"],
                params["intrinsics"]["fy"],
                params["intrinsics"]["cx"],
                params["intrinsics"]["cy"])
    
    tags = detector.detect(image, True, cam_vals, params["tag_size"])

    if (tags is None or len(tags) == 0):
        print("No tags detected")
        return None

    return tags

def get_pose(tags: list) -> list | None:
    if (tags is None):
        print("Tags empty, returning")
        return None

    best = None
    first = True

    # pick the tag closest to the camera, for least absolute error
    for tag in tags:
        current_norm = np.linalg.norm(tag.pose_t)

        # short circuits on first run, so best doesn't have to exist yet
        if (first or current_norm < np.linalg.norm(best.pose_t)):
            first = False
            best = tag

    return best.pose_t

def main():
    params = load_config("config.yaml")
    cap, det = init_capture_apriltags(params)
    img = capture_image(cap)
    frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    tags = detect_apriltags(img, det, params)
    for tag in tags:
        p1 = (int(tag.corners[0][0]), int(tag.corners[0][1]))
        p2 = (int(tag.corners[1][0]), int(tag.corners[1][1]))
        p4 = (int(tag.corners[3][0]), int(tag.corners[3][1]))
        cv2.line(frame, p1, p2, (255, 0, 0), 3)
        cv2.line(frame, p1, p4, (0, 255, 0), 3)


        print("tag_id " + str(tag.tag_id) + " with pose_t:")
        print(tag.pose_t)
    
    print(get_pose(tags))
    
    cv2.imwrite("img.png", frame)

if __name__ == "__main__":
    main()

