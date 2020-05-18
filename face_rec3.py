#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from imutils import paths
import face_recognition, cv2, os, time
import numpy as np
import pandas as pd

current_path = os.getcwd()

# Generate output directory
result_dir = os.path.join(current_path, "results")
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
                          
dataset_path = os.path.join(current_path, "dataset")
faces_path = os.path.join(current_path, "faces")
if not os.path.exists(faces_path):
    os.mkdir(faces_path)
input_video_path = os.path.join(current_path, "input_video.mp4")
output_video_path = os.path.join(result_dir, "output.mp4")
mod2_csv_path = os.path.join(result_dir, "mod-et-emo-002.csv")
mod3_csv_path = os.path.join(result_dir, "mod-et-emo-003.csv")
output_FPS = 2

boundary_color = (0, 255, 0)   # (Blue, Green, Red)
boundary_thickness = 2
font_color = (0, 255, 0)       # (Blue, Green, Red)
font_size = 0.8
font_thickness = 2

"""
num is a very important variable. It controls how much frames per second of the video we are extracting so
basically it is very computationally expensive and time consuming for long videos to perform frame by frame
detection and processing. Hence we can periodically skip some frames. This is controlled by num variable. Setting
num=1 would mean extract all frames. num = 2 means skip 1 frame and then extract the second and so on. num = 4
would mean skip 3 frames and then extract 1 and then again skip 3 and so on.
"""
num = 2

# Put the names of people which you want to recognize. The name must match the name of directory in dataset.
names_to_recognize = ["Rafa Nadal"]

"""
This is an extremely important variable and 0.6 is its recommended value. The algorithm calculates distance between
testing face embedding and all training face embeddings. If this distance is less than tolerance_threshold, match is
considered True otherwise False. Hence increasing this value means embeddings with larger distances will also be True and hence, other males or women
may also match (Flexible classifier). Decreasing this value means embeddings with lower distances will be True which means even
Rafa would may also not find a match (Strict classifier). Play with it but 0.6 is recommended from official
documentation and works fine here as well.
"""
tolerance_threshold = 0.6


"""
Thsis is the size of output faces. The main purpose of save same size of face even if images in training data
different resolutions and size. It can be any value like 64, 128, 256. Same result.
"""
size = 64


# In[2]:


def convert_frames_to_video(frames, output_FPS):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    height, width, channels = frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, output_FPS, (width, height))
    for i, ff in enumerate (frames):
        out.write(ff)
    out.release
    cv2.destroyAllWindows()
    return


# # Mod-001 ==> Video Splitting in frames

# In[3]:


cap = cv2.VideoCapture(input_video_path)
FPS = round(cap.get(cv2.CAP_PROP_FPS))
print("Frames Per Second: "+str(FPS))

frames = []
start = time.time()
count = 0
frame_originals = []
while (cap.isOpened()):
    ret, frame = cap.read()
    count = count + 1
    if not ret:
        break
    if count % num == 0:        
        frames.append(frame)
#         cv2.imshow("Original", frame)
#         cv2.waitKey(0)
end = time.time()
print("Total number of frames read: "+str(len(frames)))
print("Time taken in reading the frames: {} seconds".format(end-start))
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# # Mod-002 ==> Face detection per frame

# In[4]:


# Load some sample pictures and learn how to recognize them.

imagePaths = list(paths.list_images(faces_path))
sorted_paths = sorted(imagePaths)
known_faces = []
known_names = []

for (i, imagePath) in enumerate(sorted_paths):
    name = imagePath.split(os.path.sep)[-2]
    face_img = face_recognition.load_image_file(imagePath)
    e1 = face_recognition.face_encodings(face_img)
    if len(e1)==1:
        face_encoding = e1[0]    
        known_faces.append(face_encoding)
        known_names.append(name)
        
u = np.unique(np.asarray(known_names))
u_list = u.tolist()
count_trained_people = []
for i, n in enumerate (u_list):
    count_trained_people.append(known_names.count(n))
    
    
# Initialize some variables
face_locations = []
face_encodings = []
frame_number = 0
labelled = []
frame_num = []
person_code = []
person_name = []
Xs = []
Ys = []
Ws = []
Hs = []
scores = []
count = np.zeros(len(names_to_recognize), dtype=int)
start = time.time()

for i, ff in enumerate(frames):
    frame = ff.copy()
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
    for cc, encoding in enumerate (face_encodings):
        count_rep = np.zeros(len(names_to_recognize), dtype=int)
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_faces, encoding, tolerance=tolerance_threshold)
        dist = face_recognition.face_distance(known_faces, encoding)
        
        label = "Unknown"
        top, right, bottom, left = face_locations[cc]
        cv2.rectangle(frame, (left, top), (right, bottom), boundary_color, boundary_thickness)
        name = None
        
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for l in matchedIdxs:            
                name = known_names[l]
                if name in names_to_recognize:
                    idd = names_to_recognize.index(name)
                    count_rep[idd] += 1 
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

#             k = names_to_recognize.index(name)
#             s = count_rep[k]/count_trained_people[u_list.index(name)]
#             if s > score_thresh:
            
        if name in names_to_recognize:
            k = names_to_recognize.index(name)
#             dd = (np.mean(1-dist))/(np.amax(1-dist))
            dd = (dist - np.amin(dist))/(np.amax(dist) - np.amin(dist))
            s = 1-np.mean(dd)
#             print(matches)
#             print(count_rep[k])
#             print(np.max(dist) - np.mean(dist))
#             print(s)
            frame_num.append(i+1)
            person_code.append(k)
            person_name.append(name)
            scores.append(s)
            count[k] += 1
            print(name+" is recognized in "+str(i+1)+"th frame.")
            Xs.append(left)
            Ys.append(top)
            Ws.append(right-top)
            Hs.append(bottom-top)
            label = '%s (%.3f)' % (name, s)
                
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    #             # show the output image
#     cv2.imshow("Image", frame)
#     cv2.waitKey(0)
    
    labelled.append(frame)
    
end = time.time()
print("Time taken in processing {0:1.0f} frames: {1:2.3f} seconds".format(len(frames), end-start))
cv2.destroyAllWindows()
project2_dict = {'Frame': frame_num, 'X position': Xs, 'Y position': Ys, 'Width': Ws, 'Height': Hs,
                 'Person code': person_code, 'Person name': person_name, 'Score': scores}


# # Mod-003 ==> Aggregation of person detected

# In[10]:


aggregated_person_code = np.argmax(count)
print("Aggregated Person Code: "+str(aggregated_person_code))
aggregated_person_name = names_to_recognize[aggregated_person_code]
print("Most Frequent Person appeared: "+str(aggregated_person_name))
num = len(scores)
total_rep = 0
total = 0
for i in range (num):
    total = total + scores[i]*count[person_code[i]]
    total_rep = total_rep + count[person_code[i]]
    
final_code = total/total_rep
print("Final result: "+str(final_code))
print("\n")

project3_dict = {'Aggregated person code': aggregated_person_code, 'Aggregated person name': aggregated_person_name, 'Aggregated Score': final_code}


# # Mod-005 ==> Re-training of non-recognized faces

# In[6]:


imagePaths = list(paths.list_images(dataset_path))
sorted_paths = sorted(imagePaths)


for (i, imagePath) in enumerate(sorted_paths):
    name = imagePath.split(os.path.sep)[-2]
    output_img_folder = os.path.join(faces_path, str(name))
    if not os.path.exists(output_img_folder):
        os.mkdir(output_img_folder)
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    
    img = cv2.imread(imagePath, 1)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    if len(face_locations) != 1:
        print("This image is not suitable for training. Either more than 1 or no face has been detected here. Please delete this image from training data.")
        print("Image path is: "+str(imagePath)+"\n")
    else:
        (top, right, bottom, left) = face_locations[0]
        face = img [top:bottom, left:right]
        face = cv2.resize(face, (size,size))
#         cv2.imshow("Face", face)
#         cv2.waitKey(0)
        output_img_path = os.path.join(output_img_folder, str(i+1)+".jpg")
        cv2.imwrite(output_img_path, face)
cv2.destroyAllWindows()


# # Mod-006 ==> Saving the results

# In[33]:


frames_dir = os.path.join(result_dir, "original_frames")
if not os.path.exists(frames_dir):
    os.mkdir(frames_dir)

labelled_dir = os.path.join(result_dir, "labelled_frames")
if not os.path.exists(labelled_dir):
    os.mkdir(labelled_dir)
    
for i, f in enumerate (frames):
    file_name = os.path.join(frames_dir, str(i).zfill(5)+".jpg")
    cv2.imwrite(file_name, f)
    file_name = os.path.join(labelled_dir, str(i).zfill(5)+".jpg")
    cv2.imwrite(file_name, labelled[i])

# Save the video
convert_frames_to_video(labelled, output_FPS)
 
# Save CSV file
if os.path.exists(mod2_csv_path):
    os.remove(mod2_csv_path)
df = pd.DataFrame(data=project2_dict)
df.to_csv(mod2_csv_path, index=False, encoding='utf-8')

if os.path.exists(mod3_csv_path):
    os.remove(mod3_csv_path)
df = pd.DataFrame(data=project3_dict, index=[0])
df.to_csv(mod3_csv_path, index=False, encoding='utf-8')


# In[ ]:




