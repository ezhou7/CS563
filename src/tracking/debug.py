import cv2


def show_image(name: str, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 800, 800)

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
