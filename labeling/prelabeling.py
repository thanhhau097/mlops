import logging
import uuid
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from label_studio.core.settings.base import DATA_UNDEFINED_NAME
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path
from PIL import Image

from modeling.inference import MNISTInference

logger = logging.getLogger(__name__)


class MNISTClassification(LabelStudioMLBase):
    def __init__(self, model_path, **kwargs):
        """
        PYTHONPATH=/home/lionel/Desktop/MLE/mlops label-studio-ml init --force mnist_backend --script /home/lionel/Desktop/MLE/mlops/labeling/prelabeling.py
        PYTHONPATH=/home/lionel/Desktop/MLE/mlops label-studio-ml start ./mnist_backend
        """
        super(MNISTClassification, self).__init__(**kwargs)
        self.model_path = model_path
        self.model = MNISTInference(model_path)

    def predict(self, tasks):
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = get_image_local_path(image_url, image_dir=self.image_dir)
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("L")

        # Resize image to expected input shape
        pil_image = pil_image.resize((28, 28))

        result = [
            {
                "id": str(uuid.uuid4()),
                "type": "choices",
                "value": {"choices": [str(self.model.predict(pil_image))]},
                "to_name": "image",
                "from_name": "choice",
            }
        ]

        return [{"result": result, "score": 1}]

    def _get_image_url(self, task):
        image_url = task["data"].get("ocr") or task["data"].get(DATA_UNDEFINED_NAME)
        if image_url.startswith("s3://"):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip("/")
            client = boto3.client("s3")
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod="get_object", Params={"Bucket": bucket_name, "Key": key}
                )
            except ClientError as exc:
                logger.warning(f'Can"t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url
