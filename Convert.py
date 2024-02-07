# Convert TFLite model to Pytorch

# This uses the model face_detection_front.tflite from MediaPipe
# Using conda envrionment

!conda create -c pytorch -c conda-forge -n BlazeConv 'pytorch=1.6' jupyter opencv matplotlib
!pip install tflite

# convert front camera TFLite model
import os
import numpy as np
from collections import OrderedDict

# Get the weights from the TFLite file
# Load the TFLite model using the FlatBuffers library

!wget -N https://github.com/google/mediapipe/raw/master/mediapipe/models/face_detection_front.tflit

from tflite import Model

front_data = open("./face_detection_front.tflite", "rb").read()
front_model = Model.GetRootAsModel(front_data, 0)

front_subgraph = front_model.Subgraphs(0)
front_subgraph.Name()

def get_shape(tensor):
  return [tensor.Shape(i) for i in range(tensor.ShapeLength())]

# List all the tensors in the graph:

def print_graph(graph):
  for i in range(0, graph.TensorsLength()):
    tensor = graph.Tensors(i)
    print("%3d %30s %d %2d %s" % (i, tensor.Name(), tensor.Type(), tensor.Buffer(),
                                  get_shape(graph.Tensors(i))))

print_graph(front_subgraph)

# Make a look-up table that lets us get the tensor index based on the tensor name:
front_tensor_dict = {(front_subgraph.Tensors(i).Name().decode("utf8")): i 
                for i in range(front_subgraph.TensorLength())}

# grab only the tensors that represent weights and biases.
def get_parameters(graph):
  parameters = {}
  for i in range(graph.TensorsLength()):
    tensor = graph.Tensors(i)
    if tensor.Buffer() > 0:
      name = tensor.Name().decode("utf8")
      parameters[name] = tensor.Buffer()
  return parameters

front_parameters = get_parameters(front_subgraph)
len(front_parameters)

# The buffers are simply arrays of bytes. As the docs say,
"""
The data_buffer itself is an opaque container, with the assumption that the target device is
little-endian. In addition, all buitin operators assume the memory is ordered such that if
shape is [4,3,2], then index [i,j,k] maps to data_buffer[i*3*2 + j*2 + k]

for weights and biases, we need to interpret every 4 bytes as being as float. On my machine, the native 
byte ordering is already little-endian so we don't need to do anything special for that.

Found some weights and biases stored as float 16 instead of floaat32 corresponding to Type 1 instead of
0.
"""

def get_weights(model, graph, tensor_dict, tensor_name):
  i = tensor_dict[tensor_name]
  tensor = graph.Tensors(i)
  buffer = tensor.Buffer()
  shape = get_shape(tensor)
  assert(tensor.Type() == 0 or tensor.Type() == 1) # FLOAT32

  w = model.Buffers(buffer).DataAsNumpy()
  if tensor.Type() == 0:
    w = w.view(dtype=np.float32)
  elif tensor.Type() == 1:
    w = w.view(dtype=np.float16)
  return w

w = get_weights(front_model, front_subgraph, front_tensor_dict, "conv2d/Kernel")
b = get_weights(front_model, front_subgraph, front_tensor_dict, "conv2d/Bias")
print(w.shape, b.shape)

# Now we can get the weights for all the layers and copy them into out pytorch model
# Convert the weights to pytorch format
import torch
from blazeface import BlazeFace

front_net = BlazeFace()
print(front_net)

"""
Make a lookup table that maps the layer names between the two models. We're going to assume here
that the tensors will be in the same order in both models. if not, we should get an error because shapes
don't match.
"""

def get_probable_names(graph):
  probable_names = []
  for i in range(0, graph.TensorLength()):
    tensor = graph.Tensors(i)
    if tensor.Buffer() > 0 and (tensor.Type() == 0 or tensor.Type() == 1):
      probable_names.append(tensor.Name().decode("utf-8"))
  return probable_names

front_probable_names = get_probable_names(front_subgraph)

front_probable_names[:5]


def get_convert(net, probable_names):
  convert = {}
  i = 0
  for name, params in net.state_dict().items():
    convert[name] = probable_names[i]
    i +=1
  return convert

front_convert = get_convert(front_net, front_probable_names)

# copy the weights into the layers.
# Note that the ordering of the weights is different between Pytorch and TFLite, so we need to transpose
them.

Convolution weights:

# TFLite: (out_channels, kernel_height, kernel_width, in_channels)
# PyTorch: (out_channels, in_channels, kernel_height, kernel_width)

# Depthwise convolution weights:

# TFLite: (1, kernel_height, kernel_width, channels)
# Pytorh: (channels, 1, kernel_height, kernel_width)

def build_state_dict(model, graph, tensor_dict, net, convert):
  new_stte_dict = OrderDict()
  for dst, src in convert.items():
    w = get_weights(model, graph, tensor_dict, src)
    print(dst, src, w.shape, net.state_dict()[dst].shape)

    if w.ndim == 4:
      if w.shape[0] == 1:
        w = w.transpose((3,0,1,2)) # depthwise conv
      else:
        w = w.tranpose((0,3,1,2)) # regular conv

    new_state_dict[dst] = torch.from_numpy(w)
  return new_state_dict

front_stte_dict = build_state_dict(front_model, front_subgraph, front_tensor_dict, front_net, front_convert)

front_net.load_state_dict(front_state_dict, strict=True)

# All keys matched successfully

# No errors? Then the conversion was successful!

# save the checkpoint
torch.save(front_net.state_dict(), "blazeface.pth")

# convert back camera TFLite model
!wget -N https://github.com/google/mediapipe/raw/master/mediapipe/models/face_detection_back.tflite

back_data = open("./face_detection_back.tflite", "rb").read()
back_model = Model.GetRootAsModel(back_data,0)
back_subgraph = back_model.Subgraphs(0)
back_subgraph.Name()

print(print_graph(back_subgraph))

back_tensor_dict = {(back_subgraph.Tensors(i).Name().decode("utf8")): i
                  for i in range(back_subgraph.TensorLength())}

back_parameters = get_parameters(back_subgraph)
print(len(back_parameters))

w = get_weights(back_model, back_subgraph, back_tensor_dict, "conv2d/Kernel")
b = get_weights(back_model, back_subgraph, back_tensor_dict, "conv2d/Bias")
print(w.shape, b.shape)

back_net = BlazeFace(back_model=True)
print(back_net)

back_probable_names = get_probable_names(back_subgraph)
print(back_probable_names[:5])

back_convert = get_convert(back_net, back_probable_names)
back_state_dict = build_state_dict(back_model, back_subgraph, back_tensor_dict, back_net, back_convert)

back_net.load_state_dict(back_state_dict, strict=True)

torch.save(back_net.state_dict(), "blazefaceback.pth)
















