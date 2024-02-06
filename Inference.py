import numpy as np
import torch
import cv2

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_detections(img, detections, with_keypoints=True):
  fig, ax = plt.subplots(1, figsize=(10,10))
  ax.grid(False)
  ax.imshow(img)

  if isinstance(detections, torch.Tensor):
    detections = detections.cpu().numpy()

  if detections.ndim == 1:
    detections = np.expand_dims(detections, axis=0)

  print("Found %d faces" % detections.shape[0])

  for i in range(detections.shape[0]):
    # 정규화된 것을 원래 이미지로 확장
    ymin = detections[i,0] * img.shape[0]
    xmin = detections[i,1] * img.shape[1]
    ymax = detections[i,2] * img.shape[2]
    xmax = detections[i,3] * img.shape[3]

    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                              linewidth=1, edgecolor="r", facecolor="none",
                              alpha = detections[i,16]) # 16이 신뢰도??
    ax.add_patch(rect)

    if with_keypoints:
      for k in range(6):
        kp_x = detections[i, 4+k*2     ] * img.shape[1]
        kp_y = detections[i, 4+k*2  + 1] * img.shape[0]
        circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1,
                                edgecolor="lightskyblue", facecolor="none",
                                alpha = detections[i,16])
        ax.add_patch(circle)
  
  plt.show()


# load the front and back models

from blazeface import BlazeFace
# front는 128, back은 256 왜 이렇게 하는거지??
# back resize해서 해상도가 오르긴 했는데 data augmentation 효과 위한건가??
front_net = BlazeFace().to(gpu)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

#optionally change the thresholds:
front_net.min_score_thresh = 0.75
front_net.min_suppression_threshold = 0.3


# Make a prediction
# The input images should be 128x128 for the front model and 256x256 for the back model. BlazeFace will
# not automatically resize the image, you have to do this yourself!

img = cv2.imread("1face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

front_detections = front_net.predict_on_image(img)
front_detections.shape

front_detections

plot_detections(img, front_detections)

img2 = cv2.resize(img, (256,256))
back_detections = back_net.predict_on_image(img2)
back_detections.shape

plot_detections(img2, back_detections)











