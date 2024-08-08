import cv2
import os 
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# Initialize the list for storing the coordinates
rect_endpoint_tmp = []
rect_endpoint = []
image = None
# The callback function for the mouse
def draw_rectangle(event, x, y, flags, param):
    # Store the initial point
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_endpoint_tmp[:] = [(x, y)]
    # Store the final point when mouse is released
    elif event == cv2.EVENT_LBUTTONUP:
        rect_endpoint_tmp.append((x, y))
        cv2.rectangle(image, rect_endpoint_tmp[0], rect_endpoint_tmp[1], (0, 255, 0), 2)
        cv2.imshow('image', image)

def Reverse(tuples):
    new_tup = tuples[::-1]
    return new_tup

# def main() :
# f'{output_dir}/output_{i}.png'
root = f'checkpoints'
model_names = ['CRC_lambda_1e-05','CZC_lambda_1e-05', 'SimpleNet_lambda_1e-05','RCRC_lambda_1e-05','ZCZC_lambda_1e-05','ZPZP_lambda_1e-05']
image_names = [1,20,22,26,28,54,61,72,93,95,97,99]
if not os.path.exists(f"figures/border_test"):
    os.mkdir(f"figures/border_test")
for image_name in image_names:
    # print(f"{root}/{model_names[0]}/test_result/input_{image_name}.png")
    image = cv2.imread(f"{root}/{model_names[0]}/test_result/input_{image_name}.png")
    # clone = image.copy()
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', draw_rectangle)

    # # Keep looping until the 'q' key is pressed
    # while True:
    #     # Display the image and wait for a keypress
    #     cv2.imshow('image', image)
    #     key = cv2.waitKey(1) & 0xFF

    #     # If the 'r' key is pressed, reset the cropping region
    #     if key == ord('r'):
    #         image = clone.copy()

    #     # If the 'c' key is pressed, break from the loop (crop and save)
    #     elif key == ord('c'):
    #         rect_endpoint=rect_endpoint_tmp           
    #         break
    rect_endpoint = [[0,0],[64,64]]
    target = image[rect_endpoint[0][1]: rect_endpoint[1][1],rect_endpoint[0][0]:rect_endpoint[1][0]]
    cv2.imwrite(f"{root}/border_test/{image_name}_gt.png", target)

    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    target_tensor = transforms.ToTensor()(target)
    # print(f"{image_name}")
    
    for model_name in model_names: 
        model_img = cv2.imread(f"{root}/{model_name}/test_result/output_{image_name}.png")
        # Now we have two points for cropping the image
        model_crop_img = model_img[rect_endpoint[0][1]: rect_endpoint[1][1],rect_endpoint[0][0]:rect_endpoint[1][0]]
        model_crop_img_ = cv2.cvtColor(model_crop_img, cv2.COLOR_BGR2RGB)
        model_crop_tensor = transforms.ToTensor()(model_crop_img_)
        loss = F.huber_loss(target_tensor, model_crop_tensor, delta = 0.04)
        print(f"{image_name},{model_name},{loss}")
        
        cv2.imwrite(f"figures/border_test/{image_name}_{model_name}.png", model_crop_img)
        # print(f"figures/border_test/{image_name}_{model_name}.png saved")
# if __name__ == '__main__':
#     main()