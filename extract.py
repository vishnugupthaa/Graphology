import numpy as np
import cv2
import math

ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000
MIN_HANDWRITING_HEIGHT_PIXEL = 20

# Features are defined here as global variables
BASELINE_ANGLE = 0.0
TOP_MARGIN = 0.0
LETTER_SIZE = 0.0
LINE_SPACING = 0.0
WORD_SPACING = 0.0
PEN_PRESSURE = 0.0
SLANT_ANGLE = 0.0

def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image, d, 50, 50)
    return image

def medianFilter(image, d):
    image = cv2.medianBlur(image, d)
    return image

def threshold(image, t):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, t, 255, cv2.THRESH_BINARY_INV)
    return image

def dilate(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def erode(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image

def straighten(image):
    global BASELINE_ANGLE

    angle_sum, contour_count = 0.0, 0

    # Apply bilateral filter
    filtered = bilateralFilter(image, 3)
    # Convert to grayscale and binarize the image by INVERTED binary thresholding
    thresh = threshold(filtered, 120)
    # Dilate the handwritten lines in the image
    dilated = dilate(thresh, (5, 100))

    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)

        # Check if the contour is a line
        if h > w or h < MIN_HANDWRITING_HEIGHT_PIXEL:
            continue

        # Extract the region of interest
        roi = image[y:y+h, x:x+w]

        # Ignore short lines that may yield inaccurate baseline angles
        if w < image.shape[1] / 2:
            image[y:y+h, x:x+w] = 255
            continue

        # Compute the minimum area rectangle
        rect = cv2.minAreaRect(ctr)
        angle = rect[2]
        if angle < -45.0:
            angle += 90.0

        # Get the rotation matrix
        rot_matrix = cv2.getRotationMatrix2D((x+w/2, y+h/2), angle, 1.0)
        # Apply the affine transformation (warpAffine) to straighten the contour
        extract = cv2.warpAffine(roi, rot_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        # Overwrite the image with the straightened contour
        image[y:y+h, x:x+w] = extract

        angle_sum += angle
        contour_count += 1

    # Calculate the mean angle of the contours
    if contour_count != 0:
        mean_angle = angle_sum / contour_count
        BASELINE_ANGLE = mean_angle

    return image

''' function to calculate horizontal projection of the image pixel rows and return it '''

def horizontalProjection(image):
    height, width = image.shape[:2]
    row_sums = [image[row, :].sum() for row in range(height)]
    return row_sums


''' function to calculate vertical projection of the image pixel columns and return it '''


def verticalProjection(image):
    height, width = image.shape[:2]
    column_sums = [image[:, col].sum() for col in range(width)]
    return column_sums


''' function to extract lines of handwritten text from the image using horizontal projection '''


def extractLines(img):
    global LETTER_SIZE, LINE_SPACING, TOP_MARGIN

    # Apply bilateral filter and binarize the image
    filtered = bilateralFilter(img, 5)
    thresh = threshold(filtered, 160)

    # Calculate horizontal projection
    hpList = horizontalProjection(thresh)

    # Extract 'Top Margin' feature
    topMarginCount = sum(1 for proj in hpList if proj <= 255)

    # Extract lines using horizontal projection
    lines, space_zero, lineTop, setLineTop, setSpaceTop, includeNextSpace = [], [], 0, True, True, True
    for i, proj in enumerate(hpList):
        if proj == 0:
            if setSpaceTop:
                spaceTop, setSpaceTop = i, False
            if i < len(hpList) - 1 and hpList[i + 1] == 0:
                continue
            space_zero.append(i - spaceTop if includeNextSpace else space_zero.pop() + i - lineTop)
            setSpaceTop = True
        else:
            if setLineTop:
                lineTop, setLineTop = i, False
            if i < len(hpList) - 1 and hpList[i + 1] > 0:
                continue
            if i - lineTop >= 20:
                lines.append([lineTop, i])
                includeNextSpace = True
            else:
                includeNextSpace = False
            setLineTop = True

    # Extract fine lines
    fineLines = []
    for line in lines:
        anchor, upHill, downHill, anchorPoints = line[0], True, False, []
        for proj in hpList[line[0]:line[1]]:
            if upHill and proj >= ANCHOR_POINT:
                anchorPoints.append(anchor)
                upHill, downHill = False, True
            elif downHill and proj <= ANCHOR_POINT:
                anchorPoints.append(anchor)
                upHill, downHill = True, False
            anchor += 1
        if len(anchorPoints) < 2:
            continue
        for j in range(1, len(anchorPoints) - 1, 2):
            lineMid = (anchorPoints[j] + anchorPoints[j + 1]) // 2
            if lineMid - line[0] >= 20:
                fineLines.append([line[0], lineMid])
            line[0] = lineMid
        if line[1] - line[0] >= 20:
            fineLines.append([line[0], line[1]])

    # Calculate LINE SPACING and LETTER SIZE
    space_nonzero_row_count, midzone_row_count, lines_having_midzone_count = 0, 0, 0
    for line in fineLines:
        segment = hpList[line[0]:line[1]]
        has_midzone = any(proj >= MIDZONE_THRESHOLD for proj in segment)
        space_nonzero_row_count += sum(1 for proj in segment if proj < MIDZONE_THRESHOLD)
        midzone_row_count += sum(1 for proj in segment if proj >= MIDZONE_THRESHOLD)
        lines_having_midzone_count += has_midzone

    lines_having_midzone_count = max(lines_having_midzone_count, 1)
    total_space_row_count = space_nonzero_row_count + sum(space_zero[1:-1])
    average_line_spacing = total_space_row_count / lines_having_midzone_count
    average_letter_size = midzone_row_count / lines_having_midzone_count
    LETTER_SIZE = average_letter_size
    LINE_SPACING = average_line_spacing / average_letter_size
    TOP_MARGIN = topMarginCount / average_letter_size

    return fineLines

''' function to extract words from the lines using vertical projection '''


def extractWords(image, lines):
    global LETTER_SIZE, WORD_SPACING

    # Apply bilateral filter and binarize the image
    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 180)

    width = thresh.shape[1]
    space_zero = []  # stores the amount of space between words
    words = []  # a 2D list storing the coordinates of each word: y1, y2, x1, x2

    for line in lines:
        extract = thresh[line[0]:line[1], 0:width]
        vp = verticalProjection(extract)

        space_zero_line, word_start, space_start, index = [], None, None, 0
        for j, proj in enumerate(vp):
            if proj == 0:
                if space_start is None:
                    space_start = index
                index += 1
                if j < len(vp) - 1 and vp[j + 1] == 0:
                    continue
                if (index - space_start) > int(LETTER_SIZE / 2):
                    space_zero_line.append(index - space_start)
                space_start = None
            else:
                if word_start is None:
                    word_start = index
                index += 1
                if j < len(vp) - 1 and vp[j + 1] > 0:
                    continue

                word_end = index
                count = sum(1 for k in range(line[1] - line[0])
                            if np.sum(thresh[line[0] + k:line[0] + k + 1, word_start:word_end]))
                if count > int(LETTER_SIZE / 2):
                    words.append([line[0], line[1], word_start, word_end])

                word_start = None

        space_zero.extend(space_zero_line[1:-1])

    space_columns = sum(space_zero)
    space_count = len(space_zero) or 1
    average_word_spacing = float(space_columns) / space_count
    WORD_SPACING = average_word_spacing / LETTER_SIZE

    return words


''' function to determine the average slant of the handwriting '''


def extractSlant(img, words):
    global SLANT_ANGLE

    # Defining theta values for various slant angles
    theta = [-0.785398, -0.523599, -0.261799, -0.0872665, 0.01, 0.0872665, 0.261799, 0.523599, 0.785398]

    s_function = [0.0] * len(theta)
    count_ = [0] * len(theta)

    # Apply bilateral filter and thresholding
    filtered = bilateralFilter(img, 5)
    thresh = threshold(filtered, 180)

    for i, angle in enumerate(theta):
        s_temp = 0.0
        count = 0

        for word in words:
            original = thresh[word[0]:word[1], word[2]:word[3]]
            height, width = original.shape

            shift = (math.tan(angle) * height) / 2
            pad_length = abs(int(shift))
            new_image = np.zeros((height, width + pad_length * 2), np.uint8)
            new_image[:, pad_length:width + pad_length] = original

            pts1 = np.float32([[width / 2, 0], [width / 4, height], [3 * width / 4, height]])
            pts2 = np.float32([[width / 2 + shift, 0], [width / 4 - shift, height], [3 * width / 4 - shift, height]])
            M = cv2.getAffineTransform(pts1, pts2)
            deslanted = cv2.warpAffine(new_image, M, (width, height))

            vp = verticalProjection(deslanted)

            for k, vp_sum in enumerate(vp):
                if vp_sum == 0:
                    continue

                num_fgpixel = vp_sum / 255
                if num_fgpixel < int(height / 3):
                    continue

                column = deslanted[:, k].flatten()
                top = next((i for i, pixel in enumerate(column) if pixel > 0), height)
                bottom = next((i for i, pixel in enumerate(column[::-1]) if pixel > 0), height)
                delta_y = height - (top + bottom)

                h_sq = (float(num_fgpixel) / delta_y) ** 2
                h_wted = (h_sq * num_fgpixel) / height
                s_temp += h_wted
                count += 1

        s_function[i] = s_temp
        count_[i] = count

    max_index = s_function.index(max(s_function))

    slant_dict = {
        0: (45, " : Extremely right slanted"),
        1: (30, " : Above average right slanted"),
        2: (15, " : Average right slanted"),
        3: (5, " : A little right slanted"),
        5: (-5, " : A little left slanted"),
        6: (-15, " : Average left slanted"),
        7: (-30, " : Above average left slanted"),
        8: (-45, " : Extremely left slanted"),
        4: (0, " : No slant"),
    }

    if max_index in slant_dict:
        angle, result = slant_dict[max_index]
        if max_index == 4:
            p, q = s_function[4] / s_function[3], s_function[4] / s_function[5]
            if not (1.2 < p < 1.4 and 1.2 < q < 1.4):
                max_index = 9
                angle = 180
                result = " : Irregular slant behaviour"
            else:
                angle = 0
                result = " : No slant"

            print("\n************************************************")
            print(f"Slant determined to be {result.lower().strip()}.")
            cv2.imshow("Check Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if input("Press enter if okay, else enter c to change: ") == 'c':
                if angle == 0:
                    angle = 180
                    result = " : Irregular Slant"
                else:
                    angle = 0
                    result = " : Straight/No Slant"
                print(f"Set as {result}")
                print("************************************************\n")
            else:
                print("No Change!")
                print("************************************************\n")
    else:
        angle, result = 0, " : No slant"

    SLANT_ANGLE = angle
    print(f"Slant angle (degree): {SLANT_ANGLE}{result}")
    return


''' function to extract average pen pressure of the handwriting '''


def barometer(image):
    global PEN_PRESSURE

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image using vectorized operations for efficiency
    inverted_image = 255 - gray_image

    # Apply bilateral filter
    filtered_image = bilateralFilter(inverted_image, 3)

    # Apply binary thresholding using THRESH_TOZERO
    _, thresh_image = cv2.threshold(filtered_image, 100, 255, cv2.THRESH_TOZERO)

    # Calculate the total intensity and the number of non-zero pixels
    non_zero_pixels = thresh_image[thresh_image > 0]
    total_intensity = np.sum(non_zero_pixels)
    pixel_count = len(non_zero_pixels)

    # Calculate the average intensity
    average_intensity = total_intensity / pixel_count if pixel_count > 0 else 0

    PEN_PRESSURE = average_intensity

    return


''' main '''


def start(file_name):
    global BASELINE_ANGLE
    global TOP_MARGIN
    global LETTER_SIZE
    global LINE_SPACING
    global WORD_SPACING
    global PEN_PRESSURE
    global SLANT_ANGLE

    # Read the image from the specified file
    image = cv2.imread('images/' + file_name)
    
    # Extract pen pressure
    barometer(image)

    # Straighten the image by adjusting the contours
    straightened = straighten(image)

    # Extract lines of handwritten text using horizontal projection
    lineIndices = extractLines(straightened)

    # Extract words from each line using vertical projection
    wordCoordinates = extractWords(straightened, lineIndices)

    # Extract the average slant angle of all the words
    extractSlant(straightened, wordCoordinates)

    # Round the extracted features for consistency
    BASELINE_ANGLE = round(BASELINE_ANGLE, 2)
    TOP_MARGIN = round(TOP_MARGIN, 2)
    LETTER_SIZE = round(LETTER_SIZE, 2)
    LINE_SPACING = round(LINE_SPACING, 2)
    WORD_SPACING = round(WORD_SPACING, 2)
    PEN_PRESSURE = round(PEN_PRESSURE, 2)
    SLANT_ANGLE = round(SLANT_ANGLE, 2)

    # Return the extracted features
    return [BASELINE_ANGLE, TOP_MARGIN, LETTER_SIZE, LINE_SPACING, WORD_SPACING, PEN_PRESSURE, SLANT_ANGLE]
