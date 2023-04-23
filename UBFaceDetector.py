'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    xml_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    class_obj = cv2.CascadeClassifier(xml_path)
    # print("Starting Detection...")
    for file in os.listdir(input_path):

        abs_path = os.path.join(input_path, file)
        image = cv2.imread(abs_path)
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        objects = class_obj.detectMultiScale(grayscale_img)

        for object in objects:
            element = {
                "iname": file,
                "bbox": [
                        int(object[0]),
                        int(object[1]),
                        int(object[2]),
                        int(object[3])
                    ]
                }
            result_list.append(element)

    return result_list


'''
K: number of clusters
'''

def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []

    encoding_list = []
    img_list = []
    K = int(K)
    for imgs in os.listdir(input_path):
        img = face_recognition.load_image_file(os.path.join(input_path,imgs))
        face_locations = face_recognition.face_locations(img)
        for (t,r,b,l) in face_locations:
            # enc = face_recognition.face_encodings(img,vectors)
            encoding_list.append(np.float32(get_face_encodings[0]))
            img_list.append(imgs)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    _, labels, (centers) = cv2.kmeans(np.array(encoding_list), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    output_list = []
    for i in range(K):
        for j in range(len(labels)):
            if i == labels[j][0]:
                output_list.append(img_list[j])
        ele = {"cluster_no": i, "elements": output_list}
        output_list = []
        result_list.append(ele)


    return result_list
    

'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""

def get_face_encodings(input_path: str, faces: list) -> list:
    vectors = []
    for count,face in enumerate(faces):
        # print("starting ",count)
        # print(face)
        boxes = [(
            face["bbox"][1],
            face["bbox"][0],
            face["bbox"][1]+face["bbox"][3],
            face["bbox"][0]+face["bbox"][2],
        )]
        # print(boxes)
        image = cv2.imread(os.path.join(input_path, face["iname"]))
        # cv2.rectangle(image, (boxes[0][3], boxes[0][0]), (boxes[0][1], boxes[0][2]), (255, 0, 0), 2)
        # show_image(image,5000)
        # grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print(face_recognition.face_encodings(image, boxes))
        vectors.append((face_recognition.face_encodings(image, boxes))[0])
        # print("ending ",count)
        # print("vector appended? ",len(vectors))
    # print("detection ", len(faces),len(vectors))
    return vectors

'''
def classify(vectors: list, K: int) -> list:
    original = list(vectors)
    groups = list()
    #get K random numbers within len(vectors) range
    centroids = list(np.random.default_rng().choice(len(vectors)-1, size=int(K), replace=False))
    # centroids = [5, 19, 15, 0, 33]
    centroids.sort(reverse=True)
    #use the random numbers to initialize clusters
    groups = [[list(vectors.pop(i))] for i in centroids]
    for one in vectors:
        nearest_dist = 999999
        nearest_index = -1
        for count,i in enumerate(groups):
            # distance = get_distance(one,i[0])
            # print(count,i[0])
            if(get_distance(one,i[0]) < nearest_dist):
                nearest_dist = get_distance(one,i[0])
                nearest_index = count
                # print("replacing nearest index to ->",count)
        if nearest_index != -1:
            # print(nearest_index)
            # print("before",groups[count])
            groups[nearest_index].append(list(one))
            # print("after",groups[count])
    print(len(groups),"==>",len(groups[0]),len(groups[1]),len(groups[2]),len(groups[3]),len(groups[4]))
    # print(len(groups),"==>",len(groups[0])+len(groups[1])+len(groups[2])+len(groups[3])+len(groups[4]))
    for i in range(0,200):
        groups = kmeans_implement(groups)
    #     print(len(groups),"==>",len(groups[0]),len(groups[1]),len(groups[2]),len(groups[3]),len(groups[4]))
    # print(len(groups),"==>",len(groups[0])+len(groups[1])+len(groups[2])+len(groups[3])+len(groups[4]))
    return groups

def kmeans_implement(clusters):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    # groups = list()
    means = [ np.mean(cluster,axis=0) for cluster in clusters ]
    groups = [[] for i in means]
    for each_cluster in clusters:
        for each_img in each_cluster:
            nearest_dist = 999999
            nearest_index = -1
            for count,each_mean in enumerate(means):
                if(get_distance(each_img,each_mean) < nearest_dist):
                    nearest_dist = get_distance(each_img,each_mean)
                    nearest_index = count
            if nearest_index != -1:
                groups[nearest_index].append(list(each_img))
    return groups
    

def get_distance(vector1, vector2):
    # print(vector1,vector2)
    return (sum([(i-vector2[count])**2 for count,i in enumerate(vector1)]))**0.5

# def get_distance_group(vector1, vector2):
#     # print(vector1,vector2)
#     return (sum([(i-vector2[0][count])**2 for count,i in enumerate(vector1)]))**0.5

def get_image_clusters(faces,vectors,clusters):
    # print(len(faces),len(vectors))
    results = [ {"cluster_no": count, "elements":[]} for count,i in enumerate(clusters)]
    for count,i in enumerate(faces):
        cluster_index = get_cluster_index(list(vectors[count]),clusters)
        results[cluster_index]["elements"].append(i["iname"])
        # print(cluster_index)
    print(results,faces[0])
    return results


def get_cluster_index(vector,clusters):
    for count,i in enumerate(clusters):
        for j in i:
            if vector == j:
                return count
    return 0
'''