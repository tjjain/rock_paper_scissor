import cv2
import numpy as np
from keras.models import load_model
from random import choice

class_dir = {
	'rock':0,
	'paper':1,
	'scissor':2,
	'none':3
}

opt = ['rock','paper','scissor']

def rev_class():
	dir = {}
	m = len(class_dir.keys())
	k = list(class_dir.keys())
	for i in range(m):
		dir[i] = k[i]

	return dir

rev_class_dir = rev_class()

model = load_model('./rock_paper_scissor_model_new_1.h5')

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissor":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissor":
            return "Computer"

    if move1 == "scissor":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

###############################################################################

cap = cv2.VideoCapture(0)

while True:
	ret,frame = cap.read()
	
	if ret == False:
		continue

	cv2.rectangle(frame,(150,150),(450,450),255,1)
	cv2.imshow('video',frame)

	roi_1 = frame[150:400,150:400]
	dim = (100,100)

	roi_1 = cv2.resize(roi_1,dim)

	img_1 = np.array(roi_1)

	img_1 = img_1.reshape((-1,100,100,3))
	pred_1 = model.predict(img_1)
	pred_1 = np.argmax(pred_1)

	computer_move = choice(opt)
	if rev_class_dir[pred_1] == 'none':
		print('Waiting.....')
	else:
		print(rev_class_dir[pred_1]+" "+computer_move)
		winner = calculate_winner(rev_class_dir[pred_1],computer_move)
		print(winner)

	key_press = cv2.waitKey(1000) & 0xFF
	if key_press == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()