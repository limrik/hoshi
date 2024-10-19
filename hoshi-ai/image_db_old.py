import lancedb
from lancedb.pydantic import Vector, LanceModel
from transformers import AutoProcessor, SiglipModel
from PIL import Image
from typing import List
from io import BytesIO
import base64
import torch
import torch.nn.functional as F
import cv2
from torchvision.transforms import ToTensor
from PIL import ImageDraw
import numpy as np
from typing import Tuple
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
            similarity = (score + 1) / 2  # Original range is -1 to 1
        elif method_name == "Chi-Square":
            similarity = score  # cannot be normalized
        elif method_name == "Intersection":
            min_hist_sum = min(np.sum(hist1), np.sum(hist2))
            similarity = score / min_hist_sum if min_hist_sum != 0 else 0  # Normalize by total count
        elif method_name == "Bhattacharyya":
            similarity = 1 - score

        master_score += similarity
        # print(method_name, similarity)
    return master_score / (len(methods) + 1)


def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


class ImageItem(LanceModel):
    image_src: str
    vector: Vector(768)
    token_id: int


class ImageDB:
    def __init__(self, table_name: str):
        self.db = lancedb.connect("./.lancedb")
        self.table = self.db.create_table(table_name, schema=ImageItem, exist_ok=True)
        self.device = torch.device("mps")
        self.processor = AutoProcessor.from_pretrained("nielsr/siglip-base-patch16-224", cache_dir="models/")
        self.model = SiglipModel.from_pretrained("nielsr/siglip-base-patch16-224", cache_dir="models/", torch_dtype=torch.float16).to(self.device)
        self.batch_size = 64

    @torch.no_grad()
    def _embed_image(self, image: Image, normalized: bool = True) -> List[float]:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, dtype=torch.float16)
        image_features = self.model.get_image_features(**inputs)
        if normalized:
            image_features = F.normalize(image_features, p=2, dim=-1)[0]
        return image_features.cpu().numpy().tolist()

    def add_image(self, image_src: str, token_id: int):
        image = Image.open(image_src)
        image = image.resize((768, 1280))
        embedding = self._embed_image(image)
        self.table.add([ImageItem(image_src=image_src, vector=embedding, token_id=token_id)])

    def add_video(self, video_path: str, fps_to_use: float):
        cap = cv2.VideoCapture(video_path)
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps_to_use)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                embedding = self._embed_image(image)
                self.table.add([ImageItem(image_src=video_path, vector=embedding)])
            frame_count += 1
        cap.release()

    def delete_image(self, image_src: str = None, image_str: str = None):
        # Delete based on image_src or image_str
        if image_src:
            self.table.delete(f"image_src = '{image_src}'")
        elif image_str:
            self.table.delete(f"image_str = '{image_str}'")
        else:
            raise ValueError("Either image_src or image_str must be provided.")

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

        query_embedding = self._embed_image(image_query)
        k = 20
        # result_img_item = self.table.search(query_embedding).limit(10).to_list()[0]\
        results_df = self.table.search(query_embedding).limit(k).to_pandas()
        result_img_item = results_df.iloc[0]
        result_image_src = result_img_item["image_src"]
        result_img_embedding = torch.tensor(result_img_item["vector"]).to(self.device, dtype=torch.float16)
        query_embedding = torch.tensor(query_embedding).to(self.device, dtype=torch.float16)

        cosine_scores = torch.zeros(k)
        for i in range(k):
            score = F.cosine_similarity(query_embedding, torch.tensor(results_df.iloc[i]["vector"]).to(self.device, dtype=torch.float16), dim=-1).item()
            cosine_scores[i] = score

        whole_score = F.cosine_similarity(query_embedding, result_img_embedding, dim=-1).item()
        # print(whole_score)

        # normalize the score
        print("Original score", whole_score)

        if whole_score < 0.2:
            edited_img = image_query.copy()
            score = whole_score
        if whole_score > 1.1:  # treat as identical and draw a red outline over the entire image
            edited_img = image_query.copy()
            draw = ImageDraw.Draw(edited_img)
            draw.rectangle((0, 0, w, h), outline="red", width=3)
            score = whole_score
        else:
            heatmap, heatmap_scores = self._get_heatmap(image_query, result_img_embedding)
            heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            heatmap = cv2.resize(heatmap, (w, h))

            overlay = cv2.addWeighted(cv2.cvtColor(np.array(image_query), cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.9, 0)
            edited_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

            # irrelevant_score is the average of the 4 corners of the heatmap
            top_left_patch = heatmap_scores[0:4, 0:4]
            top_right_patch = heatmap_scores[0:4, -4:]
            bottom_left_patch = heatmap_scores[-4:, 0:4]
            bottom_right_patch = heatmap_scores[-4:, -4:]

            tl_mean = np.mean(top_left_patch)
            tr_mean = np.mean(top_right_patch)
            bl_mean = np.mean(bottom_left_patch)
            br_mean = np.mean(bottom_right_patch)

            # Compute the irrelevant score as the average of the corner patch means
            irrelevant_score = (tl_mean + tr_mean + bl_mean + br_mean) / 4

            score_from_heatmap = (heatmap_scores.max() - irrelevant_score) / (1.0 - irrelevant_score)
            # print(heatmap_scores_adj.min(), heatmap_scores_adj.flatten())
            print("Heatmap score", heatmap_scores.max(), "Irrelevant score", irrelevant_score)
            whole_score = whole_score - irrelevant_score
            whole_score = whole_score / (1.0 - irrelevant_score)
            whole_score = max(whole_score, 0.0)
            whole_score = whole_score.item()
            print("normalized score", whole_score)
            structural_score = compare_images(image_query, Image.open(result_image_src))
            print("structural score", structural_score)
            print("Score from heatmap", score_from_heatmap)
            # heatmap_scores = heatmap_scores.flatten()
            # score_from_heatmap = sum(heatmap_scores) / len(heatmap_scores)
            # print(score_from_heatmap, whole_score)
            score = whole_score * 0.6 + structural_score * 0.3 + score_from_heatmap * 0.1
            # score = 0.5 * whole_score + 0.5 * score_from_heatmap
            score = max(score, 0.0)
            score = min(score, 1.0)

            # thresh = 0.38
            # score = (score - thresh) / (1 - thresh)

        return {"image_src": result_image_src, "edited_image": edited_img, "score": score}

    @torch.no_grad()
    def _get_heatmap(self, query_image: Image, target_embedding: torch.Tensor, threshold: int = 75) -> Tuple[np.ndarray, np.ndarray]:
        num_patches_per_window = 4
        patch_height = 224 // 4
        patch_width = 224 // 4
        query_image = ToTensor()(query_image)
        patches = query_image.data.unfold(0, 3, 3)
        patches = patches.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)  # (batch, n_patches_h, n_patches_w, 3, patch_height, patch_width)

        num_patches_w = patches.shape[2]
        window = num_patches_per_window
        stride = 1
        scores = torch.zeros(patches.shape[1], patches.shape[2]).to("mps")
        runs = torch.ones(patches.shape[1], patches.shape[2]).to("mps")
        with torch.no_grad():
            # window slides from top to bottom
            for Y in range(0, patches.shape[1] - window + 1, stride):
                batch_patches = []
                batch_coords = []
                for X in range(0, patches.shape[2] - window + 1, stride):
                    patch_batch = patches[0, Y : Y + window, X : X + window]
                    big_patch = torch.zeros(window * patch_height, window * patch_width, 3)
                    for y in range(window):
                        for x in range(window):
                            big_patch[y * patch_height : (y + 1) * patch_height, x * patch_width : (x + 1) * patch_width, :] = patch_batch[y, x].permute(1, 2, 0)

                    batch_patches.append(big_patch)
                    batch_coords.append((Y, X))

                    if len(batch_patches) == self.batch_size or X == num_patches_w - window:
                        # Process the batch
                        inputs = self.processor(images=batch_patches, return_tensors="pt").to(device="mps", dtype=torch.float16)
                        image_features = self.model.get_image_features(**inputs)
                        batch_scores = F.cosine_similarity(target_embedding.unsqueeze(0), image_features, dim=-1)
                        # Update scores and runs
                        for (y, x), score in zip(batch_coords, batch_scores):
                            scores[y : y + window, x : x + window] += score
                            runs[y : y + window, x : x + window] += 1

                        # Clear the batch
                        batch_patches = []
                        batch_coords = []

        # Normalize the scores
        scores /= runs
        scores = scores.cpu().numpy()
        original_scores = scores.copy()
        scores = np.clip(scores - np.mean(scores), 0, np.inf)
        # scores = np.clip(scores - np.percentile(scores, threshold), 0, np.inf)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores, original_scores

    def search_video(self, video_path: str, fps_to_use: float, output_path: str = "output_video.mp4") -> dict:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_interval = int(original_fps / fps_to_use)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

        frame_count = 0
        extracted_frames = []
        result_counts = {}
        max_similarity = 0
        heatmap_scores = []

        # First pass: Extract frames and perform search
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                search_result = self.search_image(pil_image)
                extracted_frames.append((frame_count, pil_image, search_result))

                result_img_src = search_result["image_src"]
                result_counts[result_img_src] = result_counts.get(result_img_src, 0) + 1

                # Track the maximum similarity score
                max_similarity = max(max_similarity, search_result["score"])

            frame_count += 1

        cap.release()

        # Determine the most frequent result image
        most_common_result = max(result_counts, key=result_counts.get)
        final_result_embedding = self.table.search().where("image_src = '{}'".format(most_common_result)).limit(1).to_list()[0]["vector"]
        final_result_embedding = torch.tensor(final_result_embedding).to(self.device, dtype=torch.float16)

        # Compute heatmaps and scores for extracted frames
        heatmaps = []
        for _, pil_image, _ in extracted_frames:
            heatmap, raw_scores = self._get_heatmap(pil_image, final_result_embedding)
            heatmap = cv2.resize((heatmap * 255).astype(np.uint8), (width, height))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            heatmaps.append(heatmap)

            # Normalize and get max score for each heatmap
            normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
            heatmap_scores.append(normalized_scores.max())

        # Calculate the final score
        if max_similarity > 0.5:
            final_score = max_similarity
        else:
            whole_score = max_similarity
            avg_heatmap_score = sum(heatmap_scores) / len(heatmap_scores)
            final_score = whole_score * 0.7 + avg_heatmap_score * 0.3

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        heatmap_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                current_heatmap = heatmaps[heatmap_index]
                heatmap_index += 1
            overlay = cv2.addWeighted(frame, 0.7, current_heatmap, 0.3, 0)

            out.write(overlay)
            frame_count += 1
        cap.release()
        out.release()

        return {"img_src": most_common_result, "score": final_score, "output_video_path": output_path}


def image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def base64_to_image(image_str: str) -> Image:
    img_data = base64.b64decode(image_str)
    return Image.open(BytesIO(img_data))


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import time

    db = ImageDB("images")

    if db.table.count_rows() == 0:
        image_fpaths = os.listdir("data/images")
        image_fpaths = [os.path.join("data/images", f) for f in image_fpaths]
        image_fpaths = [x for x in image_fpaths if x.endswith(".png")]
        for image_fpath in tqdm(image_fpaths):
            db.add_image(image_fpath)
    print(db.table.count_rows())

    print("\n\nimage.png")
    image_query = Image.open("meme.png")
    start = time.time()
    results = db.search_image(image_query)
    time_taken = time.time() - start
    print(f"Found image: {results['image_src']} with score: {results['score']} in {time_taken:.2f} seconds.")
    edited_img = results["edited_image"]
    edited_img.save("edited1.png")

    # print("\n\nimage copy.png")
    # image_query = Image.open("image copy.png")
    # start = time.time()
    # results = db.search_image(image_query)
    # time_taken = time.time() - start
    # print(f"Found image: {results['image_src']} with score: {results['score']} in {time_taken:.2f} seconds.")
    # edited_img = results["edited_image"]
    # edited_img.save("edited2.png")

    # print("\n\nimage copy 2.png")
    # image_query = Image.open("image copy 2.png")
    # start = time.time()
    # results = db.search_image(image_query)
    # time_taken = time.time() - start
    # print(f"Found image: {results['image_src']} with score: {results['score']} in {time_taken:.2f} seconds.")
    # edited_img = results["edited_image"]
    # edited_img.save("edited3.png")

    # start = time.time()
    # results = db.search_video("output.mp4", fps_to_use=1.0)
    # time_taken = time.time() - start
    # print(f"Found image: {results['img_src']} with score: {results['score']} in {time_taken:.2f} seconds.")
    # print(f"Output video saved at: {results['output_video_path']}")
