from EmilBordin969367_Data_Prediction_Assignment_5 import ArthropodDatasetPredict
from EmilBordin969367_Dataset_Class_Assignment_5 import transform
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

csv_path = r"/home/jovyan/data/Annotations1-8.csv"
imgs_path = r"/home/jovyan/data/ArthroImages"
test_dataset = ArthropodDatasetPredict(csv_path, imgs_path, transform=transform)

# Load the model
model = torch.load(r"/home/jovyan/work/FasterRCNN/FasterRCNN.pt")
model.eval()

idx = 5


norm_img0, img, img_name, gtboxes, true_labels = test_dataset[idx]



to_tensor = transforms.ToTensor()
norm_img = to_tensor(norm_img0)


predictions = model(norm_img.unsqueeze(dim=0))

pboxes = predictions[0]["boxes"].detach().numpy()
predicted_labels = predictions[0]["labels"].detach().numpy()
scores = predictions[0]["scores"].detach().numpy()

score_threshold = max(scores)

pboxes = pboxes[scores >= score_threshold]
predicted_labels = predicted_labels[scores >= score_threshold]
high_scores = scores[scores >= score_threshold]
print("Prediction: ", predicted_labels)
print("True: ", true_labels)
print("Prediction scores: ", scores)


orig_ydim, orig_xdim, _ = img.shape
cropped_ydim, cropped_xdim, _ = norm_img0.shape


insect_classes = np.array(["Araneae", "Coleoptera", "Diptera", "Hemiptera", "Hymenoptera", "Lepidoptera", "Odonata"])
colors = [(255, 0, 0), (255, 255, 255), (0, 0, 255), (0, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

x_factor = orig_xdim / cropped_xdim
y_factor = orig_ydim / cropped_ydim

for class_pred, (x0, y0, xx0, yy0), score in zip(predicted_labels, pboxes, high_scores):
  class_pred -= 1
  x, y, xx, yy = int(x0*x_factor), int(y0*y_factor), int(xx0*x_factor), int(yy0*y_factor)
  cv2.rectangle(img, (x, y), (xx, yy), colors[class_pred], 5)
  cv2.putText(img, f"{insect_classes[class_pred]} ({round(score*100)}%)", (x, y-25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 0, 0), thickness=3)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.imshow(img)
plt.show()

# Visualize the ground truth boxes within the image
vlines = []
for (x0, y0, xx0, yy0) in gtboxes:
  x, y, xx, yy = int(x0*x_factor), int(y0*y_factor), int(xx0*x_factor), int(yy0*y_factor)
  cv2.rectangle(img, (x, y), (xx, yy), (0, 255, 0), 5) # Correct label in green
  cv2.putText(img, str(insect_classes[true_labels[0][0]-1]), (x, y - 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 0, 0), thickness=3)

vlines.append(img)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.imshow(img)
plt.show()
