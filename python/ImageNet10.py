import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_array = np.zeros((frame_count, frame_h, frame_w, 3), dtype=np.uint8)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        video_array[i] = frame

    cap.release()

    return frame_count, frame_w, frame_h, video_array

preds = []
pred_times = []
if __name__ == '__main__':

    # input video
    frame_count, frame_w, frame_h, vid = load_video("../data/Nimbus3000.mp4")

    target_class = 462  # (Broom)

    #preproc
    resnet_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.cuda.device("cpu")

    # New weights with accuracy 80.858%
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device=device)

    num_correct = 0

    # perform inference
    for frame_idx in range(frame_count):

        frame = vid[frame_idx]

        # Convert to Tensor and move to GPU
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resnet_transform(frame)
        frame = frame.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            probs = resnet(frame)
            _, predicted = torch.max(probs.data, 1)  # Reverse one hot encoding
            print(f"Frame {frame_idx}: Predicted class {predicted.item()}")

        # Target is broom
        if predicted.item() == target_class:
            num_correct += 1
        preds.append(predicted.item())
        pred_times.append(num_correct)

    vid_accuracy = 100 * (num_correct / frame_count)
    print(f"Video Accuracy: {vid_accuracy:.2f}%")
    print(f"Frequency: {np.unique(preds, return_counts=True)}")

    values = np.array([29, 103, 372, 418, 462, 493, 587, 618, 731, 749, 813, 828, 840, 846, 882, 907])
    frequencies = np.array([1, 2, 1, 1, 109, 1, 2, 6, 24, 2, 3, 1, 52, 1, 2, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, frame_count+1), pred_times)
    plt.show()

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(values, frequencies, width=20, align='center', color='blue', alpha=0.7)
    plt.show()
