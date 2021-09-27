import torch
from torchvision import transforms

from modeling.model import Model


class MNISTInference:
    def __init__(self, weights_path) -> None:
        self.model = Model()
        self.model.load_state_dict(torch.load(weights_path))

    def predict(self, image):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        image = transform(image).unsqueeze(0)
        output = self.model(image)
        return output.argmax(dim=1).item()


if __name__ == "__main__":
    from PIL import Image

    print("Start inference")
    inference = MNISTInference("weights/mnist_model.pt")
    image = Image.open("./data/mnist/test/1/2.png")
    print(inference.predict(image))
