from fastapi import File, UploadFile, Form, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import json
import io
import tempfile
import random
import os
import base64
from image_db import ImageDB
from text_db import TextDB

# with open("../db/users.json") as f:
#     users = json.load(f)
with open("../db/posts.json") as f:
    posts = json.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_db = ImageDB("prod_image")
text_db = TextDB("prod_text")


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


SCORE_MIN = 0.3
WEIGHT_COEFFICIENT = 2


@app.post("/search")
async def search(text: str = Form(...), file: UploadFile = File(...), component: str = Form(...)):
    if file is None:
        return {"text": text, "file": "No file uploaded"}

    file_type = "Video" if file.content_type.startswith("video") else "Image"

    content = await file.read()
    print("Start", file_type, file.content_type)
    if file_type == "Image":
        image_query = Image.open(io.BytesIO(content))
        results = image_db.search_image(image_query)
        media_score = results["score"]
        media_src = results["image_src"]
        media_token_id = int(results["token_id"])
        edited_media = encode_image(results["edited_image"])

    elif file_type == "Video":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(content)
            temp_video_path = temp_video.name

        try:
            # Use a temporary file for the output video
            with tempfile.NamedTemporaryFile(delete=False, suffix="_output.mp4") as temp_output:
                temp_output_path = temp_output.name

            results = image_db.search_video(video_path=temp_video_path, fps_to_use=4, output_path=temp_output_path)
            media_score = results["score"]
            media_src = results["image_src"]
            media_token_id = int(results["token_id"])
            edited_media = encode_video(results["output_video_path"])

        finally:
            # Clean up temporary files
            os.unlink(temp_video_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    print("Text now")
    ignore_text = len(text.split()) < 30
    if ignore_text:
        text_score = 0.0
    else:
        text_results = text_db.search(text)
        text_source = text_results[0]["source"]
        text_score = text_results[0]["score"]
        text_token_id = int(text_results[0]["token_id"])
        text_content = text_results[0]["content"]

    # if score < SCORE_MIN, assume not similar
    if text_score < SCORE_MIN:
        text_score = 0.0
    if media_score < SCORE_MIN:
        media_score = 0.0

    if component == "Text":
        text_score *= WEIGHT_COEFFICIENT
    elif component == "Image" or component == "Video":
        media_score *= WEIGHT_COEFFICIENT
    else:
        raise ValueError(f"Unknown component: {component}")

    if media_score > text_score or ignore_text:
        parent_img = Image.open(media_src)
        return {"file": media_src, "parent_type": "media", "score": media_score / WEIGHT_COEFFICIENT, "edited_media": edited_media, "parent_img": encode_image(parent_img), "token_id": media_token_id}
    else:
        return {"parent_text": text_content, "file": text_source, "parent_type": "text", "score": text_score / WEIGHT_COEFFICIENT, "token_id": text_token_id}


@app.post("/upload")
async def upload(text: str = Form(...), file: UploadFile = File(...), token_id: int = Form(...), user_id: str = Form(...)):
    content = await file.read()
    if file.filename.endswith(".jpg") or file.filename.endswith(".png"):
        file_type = "image"
    elif file.filename.endswith(".mp4"):
        file_type = "video"
    else:
        return HTTPException(status_code=400, detail="Invalid file type")

    # save to local storage
    with open(f"../db/media/{file.filename}", "wb") as f:
        f.write(content)
    if file_type == "image":
        image = Image.open(io.BytesIO(content))
        image_db.add_image(img=image, token_id=token_id, image_src=f"../db/media/{file.filename}")
    elif file_type == "video":
        image_db.add_video(video_path=f"../db/media/{file.filename}", token_id=token_id)

    text_db.add_text(text=text, source=f"{file.filename}.txt", token_id=token_id)
    posts.append(
        {
            "token_id": token_id,
            "user_handle": user_id,
            "fpath": f"../db/media/{file.filename}",
            "caption": text,
            "liked": False,
            "liked_count": random.randint(0, 100),
        }
    )
    # save the text
    with open(f"../db/texts/{file.filename}.txt", "w") as f:
        f.write(text)
    with open("../db/posts.json", "w") as f:
        json.dump(posts, f)
    print(image_db.table.count_rows())
    return {
        "message": "Upload successful",
    }


@app.get("/posts")
async def get_posts():
    return posts
