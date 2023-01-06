import cv2
# beacause the output of network is range of 0-4,which is could not be used to visualize directly,
# so we need to transform it,according to its label the 1 is vegetation,2 is building,3 is water, 4 is road,0 is background
ALL=0
VEGETATION=1
ROAD=4
BUILDING=2
WATER=3
origin_path="./data/all_test/test/2.png"
mask_path="./predict/pre2.png"
origin=cv2.imread(origin_path) # read the original image
mask=cv2.imread(mask_path)  # read the predicted image
print(origin.shape)
print(mask.shape)

# traverse each pixel,accoording to each label,we set a RGB color to it
for i in range(origin.shape[0]):
    for j in range(origin.shape[1]):
        p_mask=mask[i][j]
        if p_mask[0] == VEGETATION:
			
            origin[i][j][0] = 159
            origin[i][j][1] = 255
            origin[i][j][2] = 84
			
        elif p_mask[0] == ROAD:
			
            origin[i][j][0] = 38
            origin[i][j][1] = 71
            origin[i][j][2] = 139
			
        elif p_mask[0] == BUILDING or p_mask[0] == 255:

            origin[i][j][0] = 34
            origin[i][j][1] = 180
            origin[i][j][2] = 238

        elif p_mask[0] == WATER:

            origin[i][j][0] = 255
            origin[i][j][1] = 191
            origin[i][j][2] = 0






cv2.imwrite("stack.png", origin)


