import shutil #library for moving files
import glob #for accessing the file paths

# globals
handseg_path = "../../../handseg-150k"

def createTestSplit():
    #create directories for test data and place images there
    #separation: 70% train data and 30% test data
    destination_data = Path(handseg_path+'/test_images')
    destination_label = Path(handseg_path+'/test_masks')

    #load all file names into a list
    file_paths = glob.glob(handseg_path+"/images/*.png")
    label_paths = glob.glob(handseg_path+"/masks/*.png")

    #define amount of file for test data set
    amount_test_data = 158000*0.3

    #for the data images
    for path in file_paths:
        if amount_test_data > 0:
            shutil.move(path, destination_data)
            amount_test_data = amount_test_data - 1
        else:
            break

    amount_test_data = 158000*0.3

    #for the labeled data
    for path in label_paths:
        if amount_test_data > 0:
            shutil.move(path, destination_label)
            amount_test_data = amount_test_data - 1
        else:
            break


def createValidationSplit():
    # create validation data
    #load all file names into a list
    file_paths = glob.glob(handseg_path+"/images/*.png")
    label_paths = glob.glob(handseg_path+"/masks/*.png")

    #define amount of file for test data set 20% of data for validation
    amount_data = len(file_paths)*0.2
    tmp = amount_data

    #for the data images
    for path in file_paths:
        if tmp > 0:
            shutil.move(path, handseg_path+"/val_images/")
            tmp -= 1
        else:
            break

    tmp = amount_data

    #for the labeled data
    for path in label_paths:
        if tmp > 0:
            shutil.move(path, handseg_path+"/val_masks/")
            tmp -= 1
        else:
            break