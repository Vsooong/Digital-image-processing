import cv2
import numpy as np
import matplotlib.pyplot as plt
filename='D:/data/dip/cat.jpg'
def transformation():
    img=cv2.imread(filename)
    rows,cols,channels=img.shape

    M = np.float32([[1, 0, 100], [0, 1, 50]])
    shift_img = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite("shift img.jpg", shift_img)


    M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    rotation_img=cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite("rotation img.jpg",rotation_img)

    scaling_img=cv2.resize(img,(2*cols,rows),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("scaling img.jpg",scaling_img)

    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250]])
    M = cv2.getAffineTransform(pts1,pts2)
    affine_img = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite("affine img.jpg",affine_img)

    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    perspective_img= cv2.warpPerspective(img,M,(cols,rows))
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(perspective_img),plt.title('Output')
    # plt.show()
    cv2.imwrite("perspective img.jpg",perspective_img)

def findCorners(img, window_size=2, k=0.04):
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    newImg = np.empty(img.shape, dtype=np.float32)
    offset = int(window_size/2)

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            newImg[y][x]=r
    return newImg


def SIFT_detector_demo():
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols, channels = img.shape
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    M = np.float32([[1, 0, 200], [0, 1, 100]])
    rotation_img = cv2.warpAffine(gray, M, (cols, rows))

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('sift_keypoints.jpg', img)

    kp1, des1=sift.compute(gray,kp)
    kp2,des2=sift.detectAndCompute(rotation_img,None)
    bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches=bf.match(des1,des2,None)
    matches=sorted(matches,key=lambda x:x.distance)


    result_image=cv2.drawMatches(gray,kp1,rotation_img,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('match_image.jpg', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cornerHarris(thresh):
    corners_window = 'Corners detected.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst=findCorners(gray)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv2.circle(img, (j, i),4, (10,15,200,), 1)
    # cv2.imshow(corners_window, img)
    # cv2.waitKey(0)
    cv2.imwrite(corners_window,img)

if __name__=="__main__":
    # transformation()
    # cornerHarris(150)
    SIFT_detector_demo()
