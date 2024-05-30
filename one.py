import numpy as np
import cv2, time, math, imutils
import cv2.aruco as aruco
from math import *
from tkinter import *
from tkinter import filedialog as fd
import os
r = 100
r*=3
print('the best image size in sm: ' + str(r * 4 / 37.938105).replace('.', ',') + ':' + str(r * 1.8 / 37.938105).replace(
    '.', ','))
print('the best image size in p: ' + str(r * 4 ).replace('.', ',') + ':' + str(r * 1.8).replace(
    '.', ','))
end = 0
while True:
    path = ''
    root = Tk()
    main_list =[]
    while path == '':
        path = fd.askdirectory()
    root.destroy()


    try:
        numbers_of_markers =  os.listdir(path)

        for i in numbers_of_markers:
            if not '.' in i:
                main_list.append([int(i)])

        main_list_i = -1
        for i in numbers_of_markers:
            main_list_i+=1
            if not '.' in i:
                int(i)
                path1 = path
                path1+='/'+i+'/'
                images_for_mark = os.listdir(path1)
                list_for_add = []
                for i in images_for_mark:            
                    if i[:2]!='b_' and i[:2]!='m_':
                        list_for_add.append(i)
                main_list[main_list_i].append(list_for_add)
        path+='/'
        break
    except Exception:
        pass
#main_list = [[20, ['pushkin.jpg', 'onegin.jpg']]]

id_list = []
for i in main_list:
    id_list.append(i[0])



smooth = 0.2
def board(img, r, color):
    cv2.line(img, (r // 50 // 2, r // 50 // 2), (int(r * 1.9) + r // 50 // 2, r // 50 // 2), color, r // 50)
    cv2.line(img, (int(r * 2.1), r // 50 // 2), (int(r * 3.95), r // 50 // 2), color, r // 50)
    cv2.line(img, (int(r * 1.9) + 1, 1), (r * 2, int(r * 0.1)), color, r // 50)
    cv2.line(img, (r * 2, int(r * 0.1)), (int(r * 2.1), r // 50 // 2), color, r // 50)

    cv2.line(img, (r // 50 // 2, int(r * 2) - r // 50 // 2), ((int(r * 1.9) + r // 50 // 2, int(r * 2) - r // 50 // 2)),
             color, r // 50)
    cv2.line(img, (int(r * 2.1), int(r * 2) - r // 50 // 2), (int(r * 3.95), int(r * 2) - r // 50 // 2), color, r // 50)
    cv2.line(img, ((int(r * 1.9) + r // 50 // 2, int(r * 2) - r // 50 // 2)), (r * 2, int(r * 1.9)), color, r // 50)
    cv2.line(img, (r * 2, int(r * 1.9)), (int(r * 2.1), int(r * 2) - r // 50 // 2), color, r // 50)

    cv2.line(img, (r * 4, int(r * 0.1)), (r * 4, int(r * 0.93)), color, r // 50)
    cv2.line(img, (r * 4, int(r * 1.07)), (r * 4, int(r * 1.9)), color, r // 50)
    cv2.line(img, (r * 4, int(r * 0.93)), (int(r * 4.1), r), color, r // 50)
    cv2.line(img, (int(r * 4.1), r), (r * 4, int(r * 1.07)), color, r // 50)

    cv2.line(img, (r * 4, int(r * 0.1)), (int(r * 3.95), r // 50 // 2), color, r // 50)
    cv2.line(img, (r * 4, int(r * 1.9)), (int(r * 3.95), (r * 2) - r // 50 // 2), color, r // 50)
    return img




for i in range(len(main_list)):
    path1 =path+str(main_list[i][0])+'/'
    print(str(main_list[i][0]))
    for j in range(len(main_list[i][1])):
        image = cv2.imread(path1+main_list[i][1][j])
        if str(type(image)) == "<class 'numpy.ndarray'>":
            print(' '*len(str(main_list[i][0]))+main_list[i][1][j] + ':')
            s_img = cv2.resize(cv2.imread(path1+main_list[i][1][j]), (int(r * 4), int(r * 1.8)), interpolation=cv2.INTER_AREA)

            black = np.full([s_img.shape[0] + r // 5, int(s_img.shape[1] * 2 + r * 2.2 + 5), 3], (0, 0, 0),
                            dtype=np.uint8)
            black[r // 10:s_img.shape[0] + r // 10, :+s_img.shape[1]] = s_img

            black = board(black, r, (128, 128, 0))

            shape = s_img.shape

            s_img = black.copy()


            try:
                black[r // 10:shape[0] + r // 10, :shape[1]] = cv2.resize(cv2.imread(path1+'m_' + str(main_list[i][1][j])),
                                                                          (int(r * 4), int(r * 1.8)),
                                                                          interpolation=cv2.INTER_AREA)
                print('\t\tmask')
            except Exception:
                black[r // 10:shape[0] + r // 10, :shape[1]] = 255
                print('\t\tmask no')

            try:
                
                big = cv2.resize(cv2.imread(path1+'b_' + str(main_list[i][1][j])),(shape[1], shape[0]),interpolation=cv2.INTER_AREA)
                big_back = np.full([shape[0] + r // 5, int(shape[1] * 2 + r * 2.2 + 5), 3], (0, 0, 0),dtype=np.uint8)
                big_back[r // 10:shape[0] + r // 10, :+shape[1]] = big

                big_back = board(big_back, r, (128, 128, 0))

                big = big_back.copy()
                cv2.imwrite(main_list[i][1][j], big)
                print('\t\tbig')
            except Exception:
                big = s_img.copy()
                print('\t\tbig no')

            
            black = board(black, r, (255, 255, 255))
            main_list[i][1][j] = [s_img, black, big]

            main_list[i].append([[0,0], [0,0], 0, 0, 0, 0, 0, 0])
        else:
            main_list[i][1] = cv2.VideoCapture(main_list[i][1])
            main_list[i].append([[0,0], [0,0], 0, 1, 0, 0, 0, 0])
r=r//3
# [[id, [image1, image2], [x,y,engle,if_img, count of spaces, count of life, num_of_img,r]]]
def paste(l_img, s_img2, coor, engle):
    s_img, black_img = s_img2
    black = black_img.copy()
    mask = imutils.rotate_bound(black, engle)
    s_img = imutils.rotate_bound(s_img, engle)

    coor = [coor[0] - s_img.shape[1] // 2, coor[1] - s_img.shape[0] // 2, coor[0] + s_img.shape[1] // 2,
            coor[1] + s_img.shape[0] // 2]

    x1cor = 0
    x2cor = 0
    y1cor = 0
    y2cor = 0

    if coor[0] <= 0:
        y1cor = -1 * coor[0] + 1
        coor[0] = 0
    if coor[1] <= 0:
        x1cor = -1 * coor[1] + 1
        coor[1] = 0

    if coor[2] >= 1920:
        y2cor = coor[2] - 1919
    if coor[3] >= 1080:
        x2cor = coor[3] - 1079

    mask = cv2.inRange(mask, (250, 250, 250), (255, 255, 255))
    fg = cv2.bitwise_or(s_img, s_img, mask=mask)

    mask = cv2.bitwise_not(mask)

    x = coor[0]
    y = coor[1]
    tmp = np.full(s_img.shape, 0, dtype=np.uint8)

    l_tmp = l_img[y:y + s_img.shape[0] - x1cor - x2cor, x:x + s_img.shape[1] - y1cor - y2cor]

    tmp[x1cor:tmp.shape[0] - x2cor, y1cor:tmp.shape[1] - y2cor] = l_tmp

    mask = mask[:tmp.shape[0], :tmp.shape[1]]
    bk = cv2.bitwise_or(tmp, tmp, mask=mask)
    tmp = cv2.bitwise_or(fg[0:bk.shape[0], :bk.shape[1]], bk)

    tmp = tmp[x1cor:s_img.shape[0] - x2cor, y1cor:s_img.shape[1] - y2cor]
    shape = tmp.shape
    l_img[y:y + shape[0], x:x + shape[1]] = tmp
    return l_img


def distance(x1, y1, x2, y2):
    c = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return c


calib_path = "G:\\programs\\Python\\my_programms\\comp_vis\\AR_Games\\my\\chines\\finish\\"
camera_matrix = np.loadtxt(calib_path + 'cameraMatrix_webcam.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'cameraDistortion_webcam.txt', delimiter=',')

R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters_create()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

name = 'magic_table'
print('camera')
# -----------------------camera---------------------------
while True:
    ret, frame = cap.read()
    cv2.imshow(name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        cap.release()
        cv2.destroyAllWindows()
        break
# -----------------------camera---------------------------
print('start')
# -----------------------start----------------------------
start = 1
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
cap = cv2.VideoCapture(0)
cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
start_marker = cv2.imread('chessboard.jpg')
while start:

    ret, frame = cap.read()
    # frame = cv2.imread('ff.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          aruco_dict,
                                                          parameters=parameters)

    # frame = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
    # frame[:,:] = (255,255,255)

    if ids is not None:

        for i in range(len(ids)):
            if ids[i] == 0:
                corners1 = corners[i][0]
                xs1, ys1 = corners1[0]
                xs2, ys2 = corners1[1]
                xs3, ys3 = corners1[2]
                xs4, ys4 = corners1[3]

                k1 = (ys1 - ys2) / (xs1 - xs2)
                b1 = ys1 - k1 * xs1

                k2 = (ys1 - ys4) / (xs1 - xs4)
                b2 = ys1 - k2 * xs1
                start = 0

    cv2.putText(frame, "Start", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow(name, start_marker)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
# -----------------------start----------------------------


cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
x1 = x2 = x3 = y1 = y2 = y3 = 0
last_time = 0
time1 = [1]
x_shift = 169
y_shift = -301
anim = 0
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
while True:
    anim += 1
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          aruco_dict,
                                                          parameters=parameters)

    for i in range(len(main_list)):
        if ids is None:

            main_list[i][2][4] += 1
            if main_list[i][2][4] > 15:
                main_list[i][2][6]=0
                main_list[i][2][0], main_list[i][2][1] = [0,0], [0,0]
        else:
            if not main_list[i][0] in ids:

                main_list[i][2][4] += 1
                if main_list[i][2][4] > 15:
                    main_list[i][2][6]=0
                    main_list[i][2][0], main_list[i][2][1] = [0,0], [0,0]
            else:
                if main_list[i][2][4] > 0:
                    main_list[i][2][5]= main_list[i][2][4]
                    main_list[i][2][4] = 0


    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[:, :] = (100, 100, 100)

    cv2.putText(frame, str('fps: '+str(int(1 / time1[len(time1) - 1]))), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    if ids is not None:

        for i in range(len(ids)):
            for j in main_list:
                if 1:
                    if ids[i] == j[0]:
                        corners1 = corners[i][0]
                        x1, y1 = corners1[0]
                        x2, y2 = corners1[1]
                        x3, y3 = corners1[2]
                        x4, y4 = corners1[3]
                        j[2][7] = round(((distance(x1,y1,x3,y3)+distance(x4,y4,x2,y2))/2)*1.45/80)

                        x = [x1, x2, x3, x4]
                        y = [y1, y2, y3, y4]
                        if x1 <= x2 and y1 <= y2:
                            c = distance(x1, y1, x2, y2)
                            a = abs(y1 - y2)
                            b = abs(x1 - x2)
                            tmp = round(b / c, 4)

                            engle = 90 - int(math.degrees(math.asin(tmp)))
                        elif x1 >= x2 and y1 <= y2:
                            c = distance(x1, y1, x2, y2)
                            a = abs(y1 - y2)
                            b = abs(x1 - x2)
                            tmp = round(b / c, 4)

                            engle =90 + int(math.degrees(math.asin(tmp)))
                        elif x1 >= x2 and y1 >= y2:
                            c = distance(x1, y1, x2, y2)
                            a = abs(y1 - y2)
                            b = abs(x1 - x2)
                            tmp = round(b / c, 4)

                            engle = 270 - int(math.degrees(math.asin(tmp)))
                        else:
                            c = distance(x1, y1, x2, y2)
                            a = abs(y1 - y2)
                            b = abs(x1 - x2)
                            tmp = round(b / c, 4)

                            engle = 270 + int(math.degrees(math.asin(tmp)))

                        x_v, y_v = int((x1 + x3) / 2), int((y1 + y3) / 2)

                        k = k2
                        b = y_v - k * x_v

                        x_cor = (b1 - b) / (k - k1)

                        x_cor = (x_cor - xs1) / (xs2 - xs1)

                        k = k1
                        b = y_v - k * x_v


                        x_tmp = (b2 - b) / (k - k2)
                        y_cor = k * x_tmp + b

                        y_cor = ((k * x_tmp + b) - ys1) / (ys4 - ys1)
                        #
                        last_pos = (j[2][0],j[2][1])
                        x_cor = int(x_cor * 1920) + y_shift
                        y_cor = int(y_cor * 1080) + x_shift
                        if last_pos != (x_cor,y_cor) and (last_pos[0][1],last_pos[1][1])!=(0,0) and j[2][5]!=0 and j[2][7]==1:
                            #
                            if distance(j[2][0][1],j[2][1][1],x_cor,y_cor)/j[2][5]>35:
                                j[2][6]+=1
                                if j[2][6]>=len(j[1]):
                                    j[2][6] = 0

                            j[2][5] = 0
                        
                        if_image = j[2][3]
                        count_space = j[2][4]
                        count_life = j[2][5]
                        num_of_img = j[2][6]
                        marker_r = j[2][7]

                        if (last_pos[0][0],last_pos[1][0])!=(0,0):

                            if engle+j[2][2]>=270: #down
                                tmp_j = -j[2][2]+360
                                engle = ((engle - tmp_j) * smooth + tmp_j)%360
                            elif -engle-j[2][2]>=270: #up
                                tmp_j =  360+j[2][2]
                                engle = (engle + tmp_j) * smooth - tmp_j
                            else:
                                engle = (engle + j[2][2]) * smooth - j[2][2]
                            j[2] = [[int((x_cor-last_pos[0][0])*smooth+last_pos[0][0]),x_cor], [int((y_cor-last_pos[1][0])*smooth+last_pos[1][0]),y_cor], -1 * engle, if_image, count_space, count_life, num_of_img,marker_r]
                        else:
                            j[2] = [[x_cor,0], [y_cor,0], -1 * engle, if_image, count_space, count_life, num_of_img,marker_r]


                    # except Exception:
                    #    pass

    cv2.imshow(name, frame)
    # print(time.clock()-last_time)
    time1.append(time.clock() - last_time)
    last_time = time.clock()
    if 1:
        for j in main_list:

            if not (j[2][0][0], j[2][1][0]) == (0, 0):
                n = 6

                for i in range(n):
                    tmp = 360 / n
                    cv2.ellipse(frame, (j[2][0][0], j[2][1][0]), (int(r) - r // 100, int(r) - r // 100), 0,
                                int(anim + i * tmp), int((anim + (i + 1) * tmp) - 10), (128, 128, 0), 5)
                    anim2 = (360 - anim) * 0.9
                    cv2.ellipse(frame, (j[2][0][0], j[2][1][0]), (int(r * 0.9) - r // 100, int(r * 0.9) - r // 100), 0,
                                int(anim2 + i * tmp), int((anim2 + (i + 1) * tmp) - 10), (128, 128, 0), 5)
                if j[2][3] == 0:
                    # frame = j[1]
                    pass
                    shape = j[1][j[2][6]][0].shape
                    final_wide = (int(shape[1]*j[2][7])//25*25)//3
                    r12 = float(final_wide) / shape[1]
                    dim = (final_wide, int(shape[0] * r12))
                    if j[2][7]==1:# уменьшаем изображение до подготовленных размеров
                        s_img_p = cv2.resize(j[1][j[2][6]][0], dim, interpolation=cv2.INTER_AREA)
                        black_p = cv2.resize(j[1][j[2][6]][1], dim, interpolation=cv2.INTER_AREA)#
                    else:
                        s_img_p = cv2.resize(j[1][j[2][6]][2], dim, interpolation=cv2.INTER_AREA)
                        black_p = cv2.resize(j[1][j[2][6]][1], dim, interpolation=cv2.INTER_AREA)#[:,s_img_p.shape[1]//2]

                    frame = paste(frame, (s_img_p,black_p), (j[2][0][0], j[2][1][0]), -1 * j[2][2])  # paste(frame,cap.read()[1],(j[2][0],j[2][1]), -1*j[2][2])
                elif j[2][3] == 1:
                    pass
                    frame = paste(frame, cv2.resize(j[1].read()[1], (int(r * 2.8), int(r * 1.8)),
                                                    interpolation=cv2.INTER_AREA), (j[2][0], j[2][1]), -1 * j[2][2])



    cv2.imshow(name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        x_shift += 1
    elif key == ord('s'):
        x_shift -= 1
    elif key == ord('d'):
        y_shift += 1
    elif key == ord('a'):
        y_shift -= 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

del (time1[0])
print(y_shift, x_shift)
print('max: ' + str(max(time1) * 1000))
print('min: ' + str(min(time1) * 1000))
print('sred: ' + str(sum(time1) / len(time1) * 1000))
print('frequency: ' + str(int(1 / (sum(time1) / len(time1)))))
