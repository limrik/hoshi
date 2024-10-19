import cv2
import timm
import torch
import torch.nn.functional as F
import lancedb
import numpy as np
from typing import Union, List, Tuple
from lancedb.pydantic import Vector, LanceModel
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToTensor


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
        self.batch_size = 64

    @torch.no_grad()
    def search_image(self, image_query: Image) -> dict:
        w, h = image_query.size

        target_size = 518
        scale = max(target_size / w, target_size / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        image_query = image_query.resize((new_w, new_h))
        print("Resized image to", new_w, new_h)
        w, h = new_w, new_h
        if image_query.mode != "RGB":
            image_query = image_query.convert("RGB")

        query_embedding = self._embed_image(image_query)[0]
        k = 20
        results_df = self.table.search(query_embedding.cpu().tolist()).limit(k).to_pandas()
        k = len(results_df)
        print(results_df)
        result_img_item = results_df.iloc[0]
        result_image_src = result_img_item["image_src"]
        result_img_embedding = torch.tensor(result_img_item["vector"]).to(self.device, dtype=torch.float16)
        cosine_scores = torch.zeros(k)
        for i in range(k):
            score = F.cosine_similarity(query_embedding, torch.tensor(results_df.iloc[i]["vector"]).to(self.device, dtype=torch.float16), dim=-1).item()
            cosine_scores[i] = score

        whole_score = cosine_scores[0].item()

        # print("Original score", whole_score, "all scores", cosine_scores)

        heatmap, heatmap_scores = self._get_heatmap(image_query, result_img_embedding)
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        heatmap = cv2.resize(heatmap, (w, h))
        overlay = cv2.addWeighted(cv2.cvtColor(np.array(image_query), cv2.COLOR_RGB2BGR), 0.2, heatmap, 1, 0)
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
        score_from_heatmap = heatmap_scores.max()
        # score_from_heatmap = (heatmap_scores.max() - irrelevant_score) / (1.0 - irrelevant_score)
        # print(heatmap_scores_adj.min(), heatmap_scores_adj.flatten())
        print("Heatmap score", heatmap_scores.max(), "Irrelevant score", irrelevant_score)
        whole_score = whole_score - irrelevant_score
        whole_score = whole_score / (1.0 - irrelevant_score)
        whole_score = max(whole_score, 0.0)
        whole_score = whole_score.item()
        print("normalized score", whole_score)
        # structural_score = compare_images(image_query, Image.open(result_image_src)) # TODO what if result_image_src is a video?
        # print("structural score", structural_score)
        print("Score from heatmap", score_from_heatmap)
        # heatmap_scores = heatmap_scores.flatten()
        # score_from_heatmap = sum(heatmap_scores) / len(heatmap_scores)
        # print(score_from_heatmap, whole_score)
        # score = whole_score * 0.6 + structural_score * 0.3 + score_from_heatmap * 0.1
        score = whole_score
        # score = 0.5 * whole_score + 0.5 * score_from_heatmap
        score = max(score, 0.0)
        score = min(score, 1.0)

        # thresh = 0.38
        # score = (score - thresh) / (1 - thresh)

        return {"image_src": result_image_src, "edited_image": edited_img, "score": score}
        # return {"image_src": result_image_src, "score": score}

    def search_video(self, video_path: str, fps_to_use: float, output_path: str = "output_video.mp4") -> dict:
        frames = extract_frames_from_video(video_path, fps_to_use)
        embeddings = self._embed_image(frames)  # embeddings shape: (num_frames, embedding_dim)

        match_counts = {}
        match_scores = {}

        k = 5  # Number of top results to consider for each frame
        for embedding in embeddings:
            # Perform the search for each frame's embedding
            results_df = self.table.search(embedding.cpu().tolist()).limit(k).to_pandas()
            # Compute cosine similarity with the retrieved embeddings
            for idx, row in results_df.iterrows():
                result_embedding = torch.tensor(row["vector"]).to(self.device, dtype=torch.float16)
                cosine_sim = F.cosine_similarity(embedding, result_embedding, dim=-1).item()
                image_src = row["image_src"]
                # Update counts and scores
                if image_src not in match_counts:
                    match_counts[image_src] = 0
                    match_scores[image_src] = []
                match_counts[image_src] += 1
                match_scores[image_src].append(cosine_sim)
        print(match_counts)
        print(match_scores)
        # Compute average similarity scores for each matched item
        avg_match_scores = {image_src: sum(scores) / len(scores) for image_src, scores in match_scores.items()}

        # Find the best matching item based on the highest average similarity score
        best_match_image_src = max(avg_match_scores, key=avg_match_scores.get)
        best_match_score = avg_match_scores[best_match_image_src]

        # Optionally, generate an output video with heatmaps
        # Compute the embedding of the best matching item
        final_result_embedding = torch.tensor(self.table.search().where(f"image_src = '{best_match_image_src}'").limit(1).to_list()[0]["vector"]).to(self.device, dtype=torch.float16)

        # Prepare to write the output video with heatmaps
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_interval = int(original_fps / fps_to_use)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

        frame_count = 0
        heatmap_index = 0

        # Compute heatmaps for all frames using the best match
        heatmaps = []
        for frame in frames:
            heatmap, _ = self._get_heatmap(frame, final_result_embedding)
            heatmap = cv2.resize((heatmap * 255).astype(np.uint8), (width, height))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
            heatmaps.append(heatmap)

        # Write the output video with heatmaps overlaid
        cap = cv2.VideoCapture(video_path)
        heatmap_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0 and heatmap_index < len(heatmaps):
                current_heatmap = heatmaps[heatmap_index]
                heatmap_index += 1

            overlay = cv2.addWeighted(frame, 0.3, current_heatmap, 1, 0)
            out.write(overlay)
            frame_count += 1
        cap.release()
        out.release()

        # Return the best match information
        return {"image_src": best_match_image_src, "score": best_match_score, "output_video_path": output_path}

    @torch.no_grad()
    def _get_heatmap(self, query_image: Image, target_embedding: torch.Tensor, threshold: int = 75) -> Tuple[np.ndarray, np.ndarray]:
        num_patches_per_window = 4
        window_size = 518
        patch_height = window_size // num_patches_per_window
        patch_width = window_size // num_patches_per_window
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

                    # batch_patches.append(big_patch)
                    batch_patches.append(Image.fromarray((big_patch.numpy() * 255).astype(np.uint8)))
                    batch_coords.append((Y, X))

                    if len(batch_patches) == self.batch_size or X == num_patches_w - window:

                        image_features = self._embed_image(batch_patches)
                        batch_scores = F.cosine_similarity(target_embedding.unsqueeze(0), image_features, dim=-1)

                        for (y, x), score in zip(batch_coords, batch_scores):
                            scores[y : y + window, x : x + window] += score
                            runs[y : y + window, x : x + window] += 1

                        batch_patches = []
                        batch_coords = []

        # Normalize the scores
        scores /= runs
        scores = scores.cpu().numpy()
        original_scores = scores.copy()
        # scores = np.clip(scores - scores.mean(), 0, np.inf)
        scores = np.clip(scores - np.percentile(scores, threshold), 0, np.inf)
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores, original_scores

    @torch.no_grad()
    def _embed_image(self, img: Union[Image.Image, List[Image.Image]], normalized: bool = True) -> torch.Tensor:  # (batch_size, 768)
        if isinstance(img, list):
            inputs = torch.stack([self.img_transforms(i.convert("RGB")) for i in img]).to(self.device, dtype=torch.float16)
        else:
            inputs = self.img_transforms(img.convert("RGB")).unsqueeze(0).to(self.device, dtype=torch.float16)
        img_features = self.model(inputs)
        if normalized:
            img_features = F.normalize(img_features, p=2, dim=-1)
        return img_features

    def add_image(self, img: Image, token_id: int, image_src: str):
        vector = self._embed_image(img).cpu()[0]
        item = ImageItem(image_src=image_src, token_id=token_id, vector=vector.numpy().tolist())
        self.table.add([item])
        return

    def add_video(self, video_path: str, token_id: int):
        frames = extract_frames_from_video(video_path, self.fps_to_use)
        embeddings = self._embed_image(frames)
        for i in range(len(embeddings)):
            item = ImageItem(image_src=video_path, token_id=token_id, vector=embeddings[i].cpu().numpy().tolist())
            self.table.add([item])
        return


if __name__ == "__main__":
    import os
    import time

    # get all .png
    data_path = "data"
    fpaths = [f for f in os.listdir(data_path) if f.endswith(".jpg") or f.endswith(".png")]
    fpaths = [os.path.join(data_path, f) for f in fpaths]
    video_fpaths = [f for f in os.listdir(data_path) if f.endswith(".mp4")]
    video_fpaths = [os.path.join(data_path, f) for f in video_fpaths]
    db = ImageDB("image_db")
    if db.table.count_rows() == 0:
        for fpath in fpaths:
            img = Image.open(fpath)
            db.add_image(img, 0, fpath)
        print(db.table.count_rows())
        # db.add_video(video_path="output1.mp4", token_id=123)
        for video_fpath in video_fpaths:
            db.add_video(video_path=video_fpath, token_id=123)
    print(db.table.count_rows())
    start = time.time()
    out = db.search_image(Image.open("/Users/weihern/Documents/Computing Projects/hoshi/hoshi-ai/7) wednesday dance-original.png"))
    print("Time taken:", time.time() - start)
    print(out["image_src"], "with score", out["score"])
    out["edited_image"].save("heatmap.png")

    print()
    print()
    start = time.time()
    out = db.search_video(video_path="/Users/weihern/Documents/Computing Projects/hoshi/hoshi-ai/1b-KSI-reaction.mp4", fps_to_use=1)
    print("Time taken:", time.time() - start)
    print(out["image_src"], "with score", out["score"])
