
import os

base_path = os.getcwd()
images_path = os.path.join(base_path,'images', 'cars')
annots_path = os.path.join(base_path, 'annotations')



base_output = 'outputs'

model_path = os.path.join(base_path,base_output, 'detector_vgg16.h5')
plot_path = os.path.join(base_path, base_output, 'draw.png')
lb_PATH = os.path.sep.join([base_path, base_output,  "lb.pickle"])
test_path = os.path.join(base_path, base_output, 'test_path.txt')
PLOTS_PATH = os.path.sep.join([base_path,base_output, "plots"])


#test_paths = os.path.join(base_output, 'test_images.txt')

#LB_PATH: Our class label binarizer file, serialized in Python’s common Pickle format


INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32
