import numpy as np
import cv2
import io
from fastapi import FastAPI, Request, status, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.params import Body
from PIL import Image
from .utils import (
    load_annotation_file,
    get_prediction,
    get_stumpline_coords,
    get_average_keypoints,
    visualize_stumpline,
)


app = FastAPI()


@app.get("/ping")
def check_status():
    return "PONG"


# @app.get("/infer_video/{link}")
# This can also be used as separate API end-point
def infer_video(link: str = ""):
    annotation_file_path = get_prediction(link)
    annotation_data = load_annotation_file(annotation_file_path)

    frame_height = annotation_data["config"]["video"]["height"]

    frames_keypoints = []

    for frames in annotation_data["frames"]:
        key_points = get_stumpline_coords(frames["detections"], frame_height)

        if key_points:
            frames_keypoints.append(key_points)
        else:
            pass

    average_keypoints = get_average_keypoints(frames_keypoints)

    return average_keypoints


@app.post("/infer")
def stumpline_visualize(file=Body(...)):
    # Converting bytes image into numpy array
    image = Image.open(io.BytesIO(file))
    pil_image = image.convert("RGB")
    np_image = np.array(pil_image)

    key_points = infer_video()
    output = visualize_stumpline(key_points, np_image)

    # converting output to bytes
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    _, encoded_image = cv2.imencode(".png", output)
    output_bytes = encoded_image.tobytes()

    return Response(
        content=output_bytes, media_type="image/png", status_code=status.HTTP_200_OK
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    for error in exc.errors():
        error["message"] = error.pop("msg")

    return JSONResponse(
        content=jsonable_encoder({"detail": exc.errors()}),
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )
