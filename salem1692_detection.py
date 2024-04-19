import numpy as np
import cv2 
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

def img_augmentation(res, obj, task):
  
  x, y, w, h = obj.xywh[0].astype(int)
  
  roi = res[int(y - h/2):int(y - h/2) + h, int(x - w/2):int(x - w/2) + w]

  if task == 'accusation':
    if num_red_accusations >= 1 and num_red_accusations < 7:
        print(f'{num_red_accusations} red')
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font
        cv2.putText(res, f"{num_red_accusations} red", (10, 30), font, 1, (0, 255, 0), 2)
    elif num_red_accusations >= 7:
        print(f'Are you a WITCH?')
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font
        cv2.putText(res, f"Are you a WITCH?", (10, 30), font, 1, (0, 255, 0), 2)
    return res
  
  elif task == 'Night':
    if night_card >= 1:  # Check for detected object
        print('Close your eyes')
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font
        cv2.putText(res, f"Close your eyes", (10, 30), font, 1, (0, 255, 0), 2)
    return res
  
  elif task == 'Conspiracy':
    if conspiracy >= 1:  # Check for detected object
        print('Choose the card from your left player')
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font
        cv2.putText(res, f"Choose the card from your left player", (10, 30), font, 1, (0, 255, 0), 2)
    return res
  
  elif task == 'witch':
    if witch >= 1:  # Check for detected object
        print('Kill them all')
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font
        cv2.putText(res, f"Kill them all", (10, 30), font, 1, (0, 255, 0), 2)
    return res
  
  elif task == 'not witch':
    if not_witch >= 1:  # Check for detected object
        print('Keep Survived')
        font = cv2.FONT_HERSHEY_SIMPLEX  # Choose a suitable font
        cv2.putText(res, f"Keep Survived", (10, 30), font, 1, (0, 255, 0), 2)
    return res
  
  elif task == 'draw_bbox':
    res = cv2.rectangle(res, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + w/2)), (255, 255, 255), 2)

  elif task == 'draw_bbox':
    res = cv2.rectangle(res, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + w/2)), (255, 255, 255), 2)

  return res

show_card_message = False  # Flag to control text overlay
start_time = 0  # Variable to store start time for message display

while cap.isOpened():

  timer = cv2.getTickCount()

  ret, frame = cap.read()
  frame = cv2.resize(frame, None, fx=1.0, fy=1.0)
  res = frame.copy()

  results = model.predict(frame, conf=0.25, show=True)
  objs = results[0].boxes.numpy()
  obj_list = model.names
  if objs.shape[0] != 0:  # Check number of detected objs > 0
    num_red_accusations = 0
    night_card = 0
    conspiracy = 0
    witch = 0
    not_witch = 0
    for obj in objs:
        detected_obj = obj_list[int(obj.cls[0])]
        if detected_obj == '3red accusation':
            num_red_accusations += 3
            if num_red_accusations >= 1:
                show_card_message = True
                start_time = cv2.getTickCount()

        elif detected_obj == '1red accusation':
            num_red_accusations += 1
            if num_red_accusations >= 1:
                show_card_message = True
                start_time = cv2.getTickCount()

        elif detected_obj == '7red accusation':
            num_red_accusations += 7
            if num_red_accusations >= 1:
                show_card_message = True
                start_time = cv2.getTickCount()

        elif detected_obj == 'Night':
            night_card += 1 
            if night_card >= 1:
                show_card_message = True
                start_time = cv2.getTickCount()
        
        elif detected_obj == 'Conspiracy':
            conspiracy += 1 
            if conspiracy >= 1:
                show_card_message = True
                start_time = cv2.getTickCount()

        elif detected_obj == 'witch':
            witch += 1 
            if witch >= 1:
                show_card_message = True
                start_time = cv2.getTickCount()
        
        elif detected_obj == 'not witch':
            not_witch += 1 
            if not_witch >= 1:
                show_card_message = True
                start_time = cv2.getTickCount()

    # Modify frame based on flag and time
        if show_card_message:
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time < 5000:  # Show message for 5 seconds
                if num_red_accusations > 1:
                    aug_img = img_augmentation(res.copy(), obj, task='accusation')
                elif night_card >= 1:
                    aug_img = img_augmentation(res.copy(), obj, task='Night')
                elif conspiracy >= 1:
                    aug_img = img_augmentation(res.copy(), obj, task='Conspiracy')
                elif witch >= 1:
                    aug_img = img_augmentation(res.copy(), obj, task='witch')
                elif not_witch >= 1:
                    aug_img = img_augmentation(res.copy(), obj, task='not witch')
                else:
                    aug_img = res.copy()  # Don't modify if only 1 accusation
                cv2.imshow("7red accusation", aug_img)
            else:
                show_card_message = False  # Reset flag after 5 seconds

        else:
        # Optionally, display unmodified frame here
            cv2.imshow("7red accusation", res)  # Show frame without text overlay
 # Show frame without text overlay

  cv2.waitKey(1)

  fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
  print(fps)
 
  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
