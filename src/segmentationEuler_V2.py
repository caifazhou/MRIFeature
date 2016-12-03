import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import skimage
from skimage import segmentation
import sys

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices), figsize=(8, 3), sharex=True, sharey=True)
    for i, slice in enumerate(slices):
       axes[i].imshow(slice)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                            bottom=0.02, left=0.02, right=0.98)
    plt.show()

def image_segmentation(in_file_name, out_file_name, show_image, numseg):
    n1_img = nib.load(in_file_name)
    img_data = n1_img.get_data()
    print(img_data.shape)

    slice = np.zeros((img_data.shape[0],img_data.shape[1], img_data.shape[2]))
    segm= np.zeros((img_data.shape[0],img_data.shape[1], img_data.shape[2]), dtype = np.int16)
    for i in range(img_data.shape[2]):
        slice[:,:, i] = img_data[:, :, i, 0]
        if (slice[i].min() >= 0):
            labels1 = segmentation.slic(slice[:,:, i], compactness=30, n_segments=numseg, multichannel=False)
            segm[:, :, i] = skimage.img_as_int(labels1)


    if (show_image):
        show_slices([slice[100], slice[110], slice[120]])
        plt.suptitle("slices")

    for i in range(img_data.shape[2]):
        img_data[:, :, i, 0] = segm[:, :, i]

    if (show_image):
        # display results
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

        ax1.imshow(img_data[:,:, 100, 0])
        ax1.axis('off')
        ax1.set_title('image 100', fontsize=20)

        ax2.imshow(img_data[:,:,110,0])
        ax2.axis('off')
        ax2.set_title('image 110', fontsize=20)

        ax3.imshow(img_data[:,:,120,0])
        ax3.axis('off')
        ax3.set_title('image 120', fontsize=20)

        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                            bottom=0.02, left=0.02, right=0.98)

        plt.show()

    save_img = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(save_img, out_file_name)
    return 0
def segRunTrain(MRIIndex, numseg, inputDirBase, outputDirBase):
    i = MRIIndex
    file_name_input = inputDirBase + "set_train/train_" + str(i) + ".nii"
    file_name_output = outputDirBase + "set_train_seg/train_" + str(i) + ".nii"
    image_segmentation(file_name_input, file_name_output, False, numseg)
    return 0

def segRunTest(MRIIndex, numseg, inputDirBase, outputDirBase):
    i = MRIIndex
    file_name_input = inputDirBase + "set_test/test_" + str(i) + ".nii"
    file_name_output = outputDirBase + "set_test_seg/test_" + str(i) + ".nii"
    image_segmentation(file_name_input, file_name_output, False, numseg)
    return 0
# main begins
# load data

# train_dataFile_size = 278
# test_dataFile_size = 138

numSeg = np.int(sys.argv[1])
mriIndex = np.int(sys.argv[2])
flag = np.int(sys.argv[3])

inputBaseDir = "../testData/testOrg/"
outputBaseDir = "../seg" + str(numSeg) + "/"

if flag == 0:
    segRunTrain(mriIndex, numSeg, inputBaseDir, outputBaseDir)
else:
    segRunTest(mriIndex, numSeg, inputBaseDir, outputBaseDir)

# numSeg = 20
# mriIndex = 1


# for i in range(1):
#     file_name_input = "../testData/testOrg/set_train/train_" + str(i+1) + ".nii"
#     file_name_output = "../testData/testSeg/set_train_seg/train_" + str(i+1) + ".nii"
#     image_segmentation(file_name_input, file_name_output, False, numSeg)

# for i in range(9, test_dataFile_size-1):
#     file_name_input = "../data/set_test/test_" + str(i+1) + ".nii"
#     # example_ni1 = os.path.join(data_path, file_name_input)
#     file_name_output = "../data/set_test_seg/test_" + str(i+1) + ".nii"
#     # save_example_ni1 = os.path.join(data_path, file_name_output)
#     image_segmentation(file_name_input,file_name_output, show_image=False)