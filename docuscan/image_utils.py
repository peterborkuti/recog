import cv2
import numpy as np

def remove_text_with_morphology(img):
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)

    return img

def remove_background(img):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (5,5,img.shape[1]-5,img.shape[0]-5)

    # applying grabCut for foreground extraction
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    return img

def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 100, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    return canny

def find_contours(img, canny):
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # con = np.zeros_like(img)
    # con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)
    # showSimpleImage(con)

    return page

def order_points(pts):
    '''Rearrange coordinates to order:
        top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()

def find_corners(img, page):
    con = np.zeros_like(img)
    # Loop over the contours.
    for c in page:
        # Approximate the contour.
        arc_length = cv2.arcLength(c, True)
        if arc_length < 1000:
            break
        epsilon = 0.02 * arc_length
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points
        if len(corners) == 4:
            break
    # cv2.drawContours(con, c, -1, (0, 255, 255), 3)
    # cv2.drawContours(con, corners, -1, (0, 255, 0), 20)
    # Sorting the corners and converting them to desired shape.
    if arc_length >= 1000 and len(corners) == 4:
        corners = sorted(np.concatenate(corners).tolist())
        corners = order_points(corners)
    else:
        corners = []

    # Displaying the corners.
    #for index, c in enumerate(corners):
        #character = chr(65 + index)
        #cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 5, cv2.LINE_AA)

    #plt.figure(figsize = (10,7))
    #plt.imshow(con)
    #plt.title('Corner Points')
    #plt.show()

    # Rearranging the order of the corner points.
    #corners = order_points(corners)

    return [arc_length >= 1000 and len(corners) == 4, corners]

def calculate_destination_corners(corners):
    (tl, tr, br, bl) = corners
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [
            [0, 0],
            [maxWidth, 0],
            [maxWidth, maxHeight],
            [0, maxHeight]]
    print('destination_conrners', destination_corners)

    return destination_corners

def transform_image(orig_img, corners, destination_corners):
    # Getting the homography.
    homography = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    maxWidth, maxHeight = destination_corners[2]
    final = cv2.warpPerspective(orig_img, np.float32(homography), (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    # cv2.imwrite('grabcutop/img22.jpg', final)

    return final

def process_image(img):
    orig_img = img.copy()
    images = []

    img = remove_text_with_morphology(img)
    images.append(img.copy())
    img = remove_background(img)
    images.append(img.copy())
    canny = edge_detection(img)
    images.append(canny.copy())
    page = find_contours(img, canny)
    ret, corners = find_corners(img, page)

    if ret:
        again = False
        destination_corners = calculate_destination_corners(corners)
        final_img = transform_image(orig_img, corners, destination_corners)
    else:
        print('No corners')
        again = True
        final_img = np.zeros_like(orig_img)

    return [again, final_img, images]
