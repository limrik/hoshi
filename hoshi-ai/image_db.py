import cv2
import timm
import torch
import torch.nn.functional as F
import lancedb
import numpy as np
from typing import Union, List
from lancedb.pydantic import Vector, LanceModel
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def compare_images(img1, img2):
    img1_grey = img1.convert("L")
    img2_grey = img2.convert("L")
    img2_grey = img2_grey.resize(img1.size)

    img1_grey_array = np.array(img1_grey)
    img2_grey_array = np.array(img2_grey)
    ssim_score = ssim(img1_grey_array, img2_grey_array)

    # SSIM returns a value between -1 and 1, where 1 means identical images
    # We'll normalize it to 0-1 range, where 0 means completely different
    normalized_ssim_score = (ssim_score + 1) / 2

    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    hist1 = calculate_histogram(img1_cv)
    hist2 = calculate_histogram(img2_cv)

    methods = [
        ("Correlation", cv2.HISTCMP_CORREL),
        # ("Chi-Square", cv2.HISTCMP_CHISQR),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        # ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA),
    ]

    master_score = normalized_ssim_score
    # print("SSIM", normalized_ssim_score)
    for method_name, method_const in methods:
        score = cv2.compareHist(hist1, hist2, method_const)
        if method_name == "Correlation":
            similarity = (score + 1) / 2
        elif method_name == "Chi-Square":
            similarity = score  # cannot be normalized
        elif method_name == "Intersection":
            min_hist_sum = min(np.sum(hist1), np.sum(hist2))
            similarity = score / min_hist_sum if min_hist_sum != 0 else 0
        elif method_name == "Bhattacharyya":
            similarity = 1 - score

        master_score += similarity
        # print(method_name, similarity)
    return master_score / (len(methods) + 1)


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_frames_from_video(video_path: str, fps_to_use: int) -> List[Image.Image]:
    extracted_frames = []
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_interval = int(original_fps / fps_to_use)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            extracted_frames.append(pil_image)
        frame_count += 1

    cap.release()
    return extracted_frames


class ImageItem(LanceModel):
    image_src: str
    token_id: int
    vector: Vector(768)


class ImageDB:
    def __init__(self, table_name: str, fps_to_use: int = 1):
        self.db = lancedb.connect("./.lancedb")
        self.table = self.db.create_table(table_name, schema=ImageItem, exist_ok=True)
        self.device = torch.device("mps")
        self.model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0).eval().to(self.device, dtype=torch.float16)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.img_transforms = timm.data.create_transform(**data_config, is_training=False)
        self.fps_to_use = fps_to_use

    @torch.no_grad()
    def search_image(self, image_query: Image) -> dict:
        w, h = image_query.size
        new_w = w
        new_h = h
        while w > 768:
            new_w = new_w // 2
        while h > 1280:
            new_h = new_h // 2
        image_query = image_query.resize((new_w, new_h))
        if image_query.mode != "RGB":
            image_query = image_query.convert("RGB")

        query_embedding = self._embed_image(image_query)[0]
        k = 20
        results_df = self.table.search(query_embedding.tolist()).limit(k).to_pandas()
        k = len(results_df)

        result_img_item = results_df.iloc[0]
        result_image_src = result_img_item["image_src"]
        result_img_embedding = torch.tensor(result_img_item["vector"]).to(self.device, dtype=torch.float16)
        query_embedding = torch.tensor(query_embedding).to(self.device, dtype=torch.float16)
        cosine_scores = torch.zeros(k)
        for i in range(k):
            score = F.cosine_similarity(query_embedding, torch.tensor(results_df.iloc[i]["vector"]).to(self.device, dtype=torch.float16), dim=-1).item()
            cosine_scores[i] = score

        whole_score = cosine_scores[0].item()

        print("Original score", whole_score, "all scores", cosine_scores)

        return {"image_src": result_image_src, "score": whole_score}

    @torch.no_grad()
    def _embed_image(self, img: Union[Image.Image, List[Image.Image]], normalized: bool = True) -> torch.Tensor:  # (batch_size, 768)
        if isinstance(img, list):
            inputs = torch.stack([self.img_transforms(i) for i in img]).to(self.device, dtype=torch.float16)
        else:
            inputs = self.img_transforms(img).unsqueeze(0).to(self.device, dtype=torch.float16)
        img_features = self.model(inputs)
        if normalized:
            img_features = F.normalize(img_features, p=2, dim=-1)
        return img_features.cpu()

    def add_image(self, img: Image, token_id: int, image_src: str):
        vector = self._embed_image(img)[0]
        item = ImageItem(image_src=image_src, token_id=token_id, vector=vector.numpy().tolist())
        self.table.add([item])
        return

    def add_video(self, video_path: str, token_id: int):
        frames = extract_frames_from_video(video_path, self.fps_to_use)
        embeddings = self._embed_image(frames)
        for i in range(len(embeddings)):
            item = ImageItem(image_src=video_path, token_id=token_id, vector=embeddings[i].numpy().tolist())
            self.table.add([item])
        return


if __name__ == "__main__":
    import os
    import time

    # get all .png
    fpaths = [f for f in os.listdir() if f.endswith(".png")]
    db = ImageDB("image_db")
    if db.table.count_rows() == 0:
        for fpath in fpaths:
            img = Image.open(fpath)
            db.add_image(img, 0, fpath)
    print(db.table.count_rows())
    start = time.time()
    print(db.search_image(Image.open("image.png")))
    print(time.time() - start)
