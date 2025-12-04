import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best_resnet18_epoch100.pt"   # your trained age model
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGE_CLASSES = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60+"
]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride,
                 groups=1, activation_layer=nn.Hardswish):
        padding = (kernel_size - 1) // 2
        if activation_layer is None:
            act = nn.Identity
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_ch),
                act()
            ]
        else:
            act = activation_layer
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                          groups=groups, bias=False),
                nn.BatchNorm2d(out_ch),
                act(inplace=True),
            ]
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_ch, squeeze_ch):
        super().__init__()
        self.fc1 = nn.Conv2d(in_ch, squeeze_ch, 1)
        self.fc2 = nn.Conv2d(squeeze_ch, in_ch, 1)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)
        return x * torch.sigmoid(scale)


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size,
                 stride, squeeze_ch, use_hs=True):
        super().__init__()
        self.use_res_connect = (stride == 1 and inp == oup)
        activation_layer = nn.Hardswish if use_hs else nn.ReLU

        layers = []
        if inp == hidden_dim:
            layers.append(
                ConvBNActivation(inp, hidden_dim, kernel_size, stride,
                                 groups=hidden_dim, activation_layer=activation_layer)
            )
        else:
            layers.append(
                ConvBNActivation(inp, hidden_dim, 1, 1,
                                 activation_layer=activation_layer)
            )
            layers.append(
                ConvBNActivation(hidden_dim, hidden_dim, kernel_size, stride,
                                 groups=hidden_dim, activation_layer=activation_layer)
            )
        layers.append(SqueezeExcitation(hidden_dim, squeeze_ch))
        layers.append(
            ConvBNActivation(hidden_dim, oup, 1, 1,
                             activation_layer=None)
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            return x + out
        else:
            return out


class AgeMobileNetV3Like(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNActivation(3, 32, 3, stride=2, activation_layer=nn.Hardswish),

            nn.Sequential(
                InvertedResidual(inp=32, hidden_dim=32, oup=16,
                                 kernel_size=3, stride=1, squeeze_ch=8),),

            nn.Sequential(
                InvertedResidual(inp=16, hidden_dim=96, oup=24,
                                 kernel_size=3, stride=2, squeeze_ch=4),
                InvertedResidual(inp=24, hidden_dim=144, oup=24,
                                 kernel_size=3, stride=1, squeeze_ch=6),),
            nn.Sequential(
                InvertedResidual(inp=24, hidden_dim=144, oup=40,
                                 kernel_size=5, stride=2, squeeze_ch=6),
           

                InvertedResidual(inp=40, hidden_dim=240, oup=40,
                                 kernel_size=5, stride=1, squeeze_ch=10),),

            nn.Sequential(
                InvertedResidual(inp=40, hidden_dim=240, oup=80,
                                 kernel_size=3, stride=2, squeeze_ch=10),
                InvertedResidual(inp=80, hidden_dim=480, oup=80,
                                 kernel_size=3, stride=1, squeeze_ch=20),
                InvertedResidual(inp=80, hidden_dim=480, oup=80,
                                 kernel_size=3, stride=1, squeeze_ch=20),),

            nn.Sequential(
                InvertedResidual(inp=80, hidden_dim=480, oup=112,
                                 kernel_size=5, stride=1, squeeze_ch=20),
                InvertedResidual(inp=112, hidden_dim=672, oup=112,
                                 kernel_size=5, stride=1, squeeze_ch=28),
                InvertedResidual(inp=112, hidden_dim=672, oup=112,
                                 kernel_size=5, stride=1, squeeze_ch=28),
            ),

            nn.Sequential(
                InvertedResidual(inp=112, hidden_dim=672, oup=192,
                                 kernel_size=5, stride=2, squeeze_ch=28),
                InvertedResidual(inp=192, hidden_dim=1152, oup=192,
                                 kernel_size=5, stride=1, squeeze_ch=48),
                InvertedResidual(inp=192, hidden_dim=1152, oup=192,
                                 kernel_size=5, stride=1, squeeze_ch=48),
                InvertedResidual(inp=192, hidden_dim=1152, oup=192,
                                 kernel_size=5, stride=1, squeeze_ch=48),
            ),

            nn.Sequential(
                InvertedResidual(inp=192, hidden_dim=1152, oup=320,
                                 kernel_size=3, stride=1, squeeze_ch=48),
            ),
            ConvBNActivation(320, 1280, 1, stride=1, activation_layer=nn.Hardswish),)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_resnet18(num_classes: int = 8) -> nn.Module:
    return AgeMobileNetV3Like(num_classes=num_classes)

def load_model():
    model = make_resnet18(num_classes=8).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# inference
def predict(image_path):
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)  

    model = load_model()

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    print("\nImage:", image_path)
    print("Predicted class:", AGE_CLASSES[pred_class])

    print("\nProbabilities:")
    for c, p in zip(AGE_CLASSES, probs):
        print(f"  {c:7s} : {p.item()*100:.2f}%")

    return pred_class

# CLI USAGE
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python test_image.py path/to/image.jpg\n")
        sys.exit(0)

    image_path = sys.argv[1]
    predict(image_path)
