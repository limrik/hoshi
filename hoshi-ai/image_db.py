import timm
import torch
import lancedb
from lancedb.pydantic import Vector, LanceModel
from PIL import Image


class ImageItem(LanceModel):
    image_src: str
    token_id: int
    vector: Vector(768)


class ImageDB:
    def __init__(self, table_name: str):
        self.db = lancedb.connect("./.lancedb")
        self.table = self.db.create_table(table_name, schema=ImageItem, exist_ok=True)
        self.device = torch.device("mps")
        self.model = (
            timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
            )
            .eval()
            .to(self.device)
        )
        data_config = timm.data.resolve_model_data_config(self.model)
        self.img_transforms = timm.data.create_transform(**data_config, is_training=False)

    @torch.no_grad()
    def _embed_image(self, img: Image) -> torch.Tensor:
        return self.model(self.img_transforms(img).unsqueeze(0).to(self.device))  # (batch_size, 768)
