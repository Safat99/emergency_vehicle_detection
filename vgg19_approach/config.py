import os

#base path of vgg16
base_path = os.path.abspath(os.path.join(os.getcwd(), '../vgg_16_approach'))
images_path = os.path.join(base_path,'images', 'cars')


base_output = 'outputs_vgg19'

model_path = os.path.join(os.getcwd(),base_output, 'detector_vgg19.h5')
plot_path = os.path.join(os.getcwd(), base_output, 'draw.png')
lb_PATH = os.path.sep.join([os.getcwd(), base_output,  "lb.pickle"])
test_path = os.path.join(os.getcwd(), base_output, 'test_path.txt')
PLOTS_PATH = os.path.sep.join([os.getcwd(), base_output, "plots"])


INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32
