import os

DATASET_ROOT = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data\Kinetics'

ANNOTATIONS = os.path.join(DATASET_ROOT, "kinetics400")
TRAIN_ROOT = os.path.join(ANNOTATIONS, "train.csv")
VALID_ROOT = os.path.join(ANNOTATIONS, "validate.csv")
TEST_ROOT = os.path.join(ANNOTATIONS, "test.csv")

FRAMES = os.path.join(DATASET_ROOT, "frames")
TRAIN_FRAMES_ROOT = os.path.join(FRAMES, "train_frames")
VALID_FRAMES_ROOT = os.path.join(FRAMES, "valid_frames")
TEST_FRAMES_ROOT = os.path.join(FRAMES, "test_frames")
