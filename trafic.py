from matplotlib import path
import numpy as np
import subprocess
import datetime
import math
import cv2
import os


fps = 15
base_path = "/home/cera/auto_trafic/"
video_path = base_path+"video/"
thresh_path = 10
thresh_end_track = 5
thresh_accuracy = 60
thresh_distance = 30

area = 2

debug = False
path_to_save = base_path+"tracks/"
show_speed = 0


def dist(x1,y1,x2,y2):
    return int(math.ceil(np.sqrt((x1-x2)**2+(y1-y2)**2)))


def test_wwc():
    if area == 1:
        zone1 = [(300, 305), (181, 396), (655, 469), (688, 346)] #14/15
        zone2 = [(429, 79), (385, 90), (485, 112), (521, 96)] #13/12
        zone3 = [(551, 112), (526, 137), (645, 154), (652, 126)] #17/16
        zone4 = [(34, 213), (120, 173), (232, 233), (154, 282)] #15/14
        zone5 = [(468, 116), (379, 161), (482, 177), (526, 128)] #3/3
        all_coords = [zone1,zone2,zone3,zone4,zone5]
    elif area == 2:
        zone1 = [(212, 199), (89, 305), (286, 373), (353, 254)] #6/6
        zone2 = [(357, 259), (290, 378), (497, 426), (514, 297)] #3/3
        all_coords = [zone1,zone2]
    elif area == 3:
        zone1 = [(159, 416), (192, 503), (263, 472), (222, 396)] #8/9
        zone2 = [(437, 494), (571, 424), (631, 493), (503, 564)] #9/11
        zone3 = [(514, 264), (577, 330), (624, 299), (563, 243)] #4/5
        zone4 = [(287, 242), (314, 294), (425, 251), (393, 207)] #10/9
        all_coords = [zone1,zone2,zone3,zone4]
    else:
        all_coords = []

    yolo_cmd = "./create_txt auto_trafic "+str(area)
    yolo = subprocess.Popen(yolo_cmd, shell = True)
    yolo.wait()

    file = video_path+str(area)+".txt"
    
    for region in all_coords:
        miss = workWithCoords(region,file)
        print(miss)

    # rm_cmd = "rm "+file
    # rm = subprocess.Popen(rm_cmd, shell = True)
    # rm.wait()


def point_in_path(coords_path,coords_point):
    p = path.Path(coords_path)
    return p.contains_point(coords_point)


def workWithCoords(region,file="coords.txt"):
    ##################################################
    if debug:
        resol = (576,720,1)
        img = np.zeros(resol, np.uint8)
    regionPoints = np.array([region], dtype=np.int32)
    ##################################################
    dictionary = {}
    images = {}
    miss = 0
    frame = 0
    ####################################################    
    try:
        with open(file,"r") as f:
            for lines in f:
                coords = [int(a) for a in lines.split("\n")[0].split(" ")]
                [left,right,top,bot,object_class,accuracy] = coords

                if object_class == 1:
                    continue
    
                x0 = (right-left)/2+left
                y0 = (bot-top)/2+top

                if left == -1 and right == -1 and top == -1 and bot == -1:
                    frame,dictionary,miss = new_frame(frame,dictionary,miss,images)

                if accuracy < thresh_accuracy:
                    continue
    ####################################################################################################################################
                if point_in_path(region,np.array([x0,y0])):
                    if dictionary == {}:
                        dictionary[1] = [x0,y0,1,x0,y0,0]

                        if debug:
                            img = np.zeros(resol, np.uint8)
                            images[1] = [img]
                    else:
                        flag = 0
                        min_dist = thresh_distance
                        
                        for key in dictionary:
                            distant = dist(dictionary[key][0],dictionary[key][1],x0,y0)
    
                            if distant < min_dist:
                                min_dist = distant
                                real_key = key
                                flag = 1

                        if flag == 1:
                            dictionary[real_key][0] = x0
                            dictionary[real_key][1] = y0
                            dictionary[real_key][2] = 1
                            dictionary[real_key][5] += min_dist

                            if debug:
                                images[real_key][0] = draw_move(images[real_key][0],x0,y0,regionPoints)

                        else:
                            for dic_key in range(1,100):
                                if dic_key in dictionary.keys():
                                    continue
                                else:
                                    break
                            dictionary[dic_key] = [x0,y0,1,x0,y0,0]

                            if debug:
                                img = np.zeros(resol, np.uint8)
                                images[dic_key] = [img]
    ########################################################################################################
    except IOError as e:
        print("No such file: " + file)

    return miss


def new_frame(frame,dictionary,miss,images):
    del_keys = []

    frame,filename = filename_in_time(frame)

    for key in dictionary:
        dictionary[key][2] += 1
        if dictionary[key][2] >= thresh_end_track:
            del_keys.append(key)

    for del_key in del_keys:
        dictionary,miss = inc_moving(dictionary,del_key,miss,images,filename)

    return frame,dictionary,miss


def filename_in_time(frame):
    frame += 1
    sec = float(frame) / fps
    minut = int(sec // 60)
    sec -= minut * 60
    return frame,str(minut)+":"+str(int(sec))


def inc_moving(dictionary,del_key,miss,images,filename_in_time):
    if dictionary[del_key][5] > thresh_path: 
        miss += 1                                    
        if debug:
            cv2.imwrite(path_to_save+filename_in_time+" "+str(miss)+'_miss.jpg', images[del_key][0])

    dictionary.pop(del_key)
    return dictionary, miss


def draw_move(img,x0,y0,regionPoints):
    cv2.drawContours(img, regionPoints, 0, 255, thickness=3)
    cv2.circle(img,(x0,y0), 1, 255, 1)

    if show_speed != 0:
        cv2.imshow('Moving', img)
        cv2.waitKey(show_speed)
    
    return img


test_wwc()