import caffe
import sys
import pickle
import numpy as np
#caffe_root = "/home/User/caffe/"
#sys.path.insert(0, caffe_root + 'python')
caffe.set_mode_gpu()


class CaffePlugin():
  def input(self, file):
    with open(file) as config:
      self.model_def = config.readline().strip()
      self.model_weights = config.readline().strip()
      data_file = config.readline().strip()
      labels_file = config.readline().strip()
      npy_file = config.readline().strip()

      image_paths = []
      for line in config:
         image_paths.append(line.strip())
    #print(npy_file)
    #print(image_paths)
    #image_paths = ['plugins/DeepLearningClassification/example/images/cat.jpg', 'plugins/DeepLearningClassification/example/images/fish-bike.jpg']
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    #mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    #print("ENTER")
    #x = input()
    mu = np.load(npy_file)
    #mu = np.load('plugins/DeepLearningClassification/example/models/ilsvrc_2012_mean.npy')
    #print("ENTER")
    #x = input()
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    batch_size = len(image_paths) # Number of images

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': (batch_size, 3, 227, 227)})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    print("SETTING SELF DATA")
    self.data = []#transformed_images = []
    for image_path in image_paths:
         image = caffe.io.load_image(image_path)
         transformed_image = transformer.preprocess('data', image)
         self.data.append(transformed_image) #transformed_images.append(transformed_image)




    #with open(data_file) as pickled_data:
    #  self.data = pickle.load(pickled_data)
    self.labels = np.loadtxt(labels_file, str, delimiter='\t')


  def run(self):
    net = caffe.Net(self.model_def,      # defines the structure of the model
                    self.model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    in_shape = list(net.blobs['data'].data.shape)
    in_shape[0] = len(self.data) # set new batch size
    net.blobs['data'].reshape(*tuple(in_shape))
    for i, data in enumerate(self.data):
      net.blobs['data'].data[i,:,:,:] = data   
    output = net.forward()
    last_layer_name = net._layer_names[-1]
    output_prob = output[last_layer_name]
    predicted_value = (output_prob.argmax(axis=1))
    self.predicted_values = self.labels[predicted_value]


  def output(self, file):
    with open(file,"w+") as f:
      f.write(str(self.predicted_values))

