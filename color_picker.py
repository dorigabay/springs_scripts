import cv2
import numpy as np
import imutils
import colorsys

image_src = None
image_hls = None  # global ;(
pixel = (20, 60, 80)  # some stupid default


# mouse callback function
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        orig = image_src[y, x]

        bgr_color = orig

        pixel = np.uint8([[[orig[0], orig[1], orig[2]]]])
        pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HLS)
        hls_color = pixel[0][0]

        pixel = np.uint8([[[orig[0], orig[1], orig[2]]]])
        pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        hsv_color = pixel[0][0]

        print("Selected BGR color: ", bgr_color)
        print("Selected HSV color: ", hsv_color)
        print("Selected HLS color: ", hls_color)

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([hls_color[0] + 10, hls_color[1] + 10, hls_color[2] + 10])
        lower = np.array([hls_color[0] - 10, hls_color[1] - 10, hls_color[2] - 10])

        image_hls = cv2.cvtColor(image_src, cv2.COLOR_BGR2HLS)
        #cv2.imshow("hls", image_hls)

        mask = cv2.inRange(image_hls, lower, upper)

        cv2.imshow("mask", mask)
        # this generates black and white image.
        # white for all those where the colors are in range otherwise black
        res = cv2.bitwise_and(image_hls, image_hls, mask=mask)

        cv2.imshow('res.png', res)


def main(VIDEO_PATH):
    import sys
    global image_hls, pixel, image_src  # so we can use it in mouse callback

    # image_src = cv2.imread(sys.argv[1])  # pick.py my.png
    image_src = cv2.imread(VIDEO_PATH)  # pick.py my.png

    if image_src is None:
        print("the image read is None............")
        return
    image_src = imutils.resize(image_src, width=1000)
    cv2.imshow("bgr", image_src)

    ## NEW ##
    cv2.namedWindow('bgr')
    cv2.setMouseCallback('bgr', pick_color)

    # now click into the hsv img , and look at values:
    image_hls = cv2.cvtColor(image_src, cv2.COLOR_BGR2HLS)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


VIDEO_PATH = "Z:/Dor_Gabay/videos/10.9/S520007.jpg"

main(VIDEO_PATH)

# if __name__ == '__main__':
#     main()
