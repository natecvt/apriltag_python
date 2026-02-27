import cv2
import numpy as np
import keyboard
from os import PathLike
import yaml

def calibrate(path: str | PathLike):

    if not path.endswith((".yaml", ".yml")):
        print("Path does not specify a config")
        exit(1)

    with open(path, "r") as stream:
        params = yaml.safe_load(stream)

        if params is None:
            print("Invalid file syntax or data")
            exit(2)

    rows = params["checkerboard"]["rows"]
    cols = params["checkerboard"]["cols"]
    nimgs = params["num_imgs"]
    tct = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    worldPtsCurrent = np.zeros((rows*cols,3), np.float32)
    worldPtsCurrent[:,:2] = np.mgrid[0:rows, 0:cols].T.reshape(-1,2)
    worldPts = []
    imgPts = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        exit(1)

    print("Camera registered, please move calibration board within camera view")
    cv2.waitKey(2000)

    counter = 0
    while not counter >= nimgs and not keyboard.is_pressed("esc"):
        ret, frame = cap.read()

        if ret == False:
            continue
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv2.findChessboardCorners(image, (rows, cols), None)
        h, w = image.shape[:2]

        if cornersFound == True:
            print("Checkerboard Found")
            worldPts.append(worldPtsCurrent)

            cv2.rectangle(frame, (0,0), (w - 1, h - 1), (0,255,0), 7)
            
            cornersRefined = cv2.cornerSubPix(image, cornersOrg, (11,11), (-1,-1), tct)
            cv2.drawChessboardCorners(frame, (rows, cols), cornersRefined, cornersFound)
            imgPts.append(cornersRefined)

            counter = counter + 1
        else:
            print("Checkerboard Not Found")
            cv2.rectangle(frame, (0,0), (w - 1, h - 1), (0,0,255), 7)

        cv2.imshow("Calibration", frame)
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()

    if not counter == nimgs:
        print("Not all calibration-ready images taken")
        exit(2)

    print("Calculating calibration")
    repErr, camMat, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        worldPts,
        imgPts,
        image.shape[::-1],
        None,
        None)
    
    print("Writing Camera Matrix Values:\n", camMat)

    params["intrinsics"]["fx"] = float(camMat[0,0])
    params["intrinsics"]["fy"] = float(camMat[1,1])
    params["intrinsics"]["cx"] = float(camMat[0,2])
    params["intrinsics"]["cy"] = float(camMat[1,2])

    params["intrinsics"]["k1"] = float(distCoeffs[0,0])
    params["intrinsics"]["k2"] = float(distCoeffs[0,1])
    params["intrinsics"]["p1"] = float(distCoeffs[0,2])
    params["intrinsics"]["p2"] = float(distCoeffs[0,3])
    params["intrinsics"]["k3"] = float(distCoeffs[0,4])

    with open(path, "w") as stream:
        yaml.dump(params, stream, sort_keys=False)

    print("Calibration Complete")

if __name__ == "__main__":
    calibrate("apriltag_python/config.yaml")