import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

skip=0
hand_data = []
dataset_path='./data/'
file_name = input("enter: ")

while True:
	ret,frame = cap.read()
    
	if ret == False:
		continue

	cv2.rectangle(frame,(150,150),(450,450),255,1)
	cv2.imshow('video',frame)

	roi_1 = frame[150:450,150:450]
	cv2.imshow('roi',roi_1)

	hand_section = cv2.resize(roi_1,(100,100))

	skip+=1
	if skip%5 == 0:
		hand_data.append(hand_section)

	if skip >= 1000:
		break

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

hand_data = np.asarray(hand_data)
hand_data = hand_data.reshape((hand_data.shape[0],-1))
print(hand_data.shape)

np.save(dataset_path+file_name+'.npy',hand_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()