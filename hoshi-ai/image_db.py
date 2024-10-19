import cv2
import timm
import torch
import lancedb
from typing import Union, List
from lancedb.pydantic import Vector, LanceModel
from PIL import Image


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
        self.model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0).eval().to(self.device)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.img_transforms = timm.data.create_transform(**data_config, is_training=False)
        self.fps_to_use = fps_to_use

    @torch.no_grad()
    def _embed_image(self, img: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:  # (batch_size, 768)
        if isinstance(img, list):
            inputs = torch.stack([self.img_transforms(i) for i in img]).to(self.device)
        else:
            inputs = self.img_transforms(img).unsqueeze(0).to(self.device)
        return self.model(inputs)

    def add_image(self, img: Image, token_id: int, image_src: str):
        vector = self._embed_image(img)[0]
        item = ImageItem(image_src=image_src, token_id=token_id, vector=vector.cpu().numpy().tolist())
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
    db = ImageDB("image_db")
    img = Image.open("image.png")
    db.add_image(img, 1, "test.jpg")
    db.add_video("output.mp4", 1)

    print(db.table.count_rows())
