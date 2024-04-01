import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,0,255), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-5)', '(6-14)', '(15-22)', '(23-27)', '(28-38)', '(39-50)', '(50-80)', '(80-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

padding=20

def detect_age_gender(image_path=None):
    video=cv2.VideoCapture(image_path if image_path else 0)
    while cv2.waitKey(1)<0:
        hasFrame,frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            break

        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)

def browse_file():
    file_path = filedialog.askopenfilename()
    detect_age_gender(file_path)

def open_webcam():
    detect_age_gender()

def quit_app():
    root.destroy()

root = tk.Tk()
root.title("Age and Gender Detection")

# Center the window
window_width = 400
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

style = ttk.Style()
style.configure('TButton', font=('Arial', 12), borderwidth=4, foreground='black', background='lightgray', padding=5)
style.configure('TLabel', font=('Arial', 12), foreground='black', background='white')
style.configure('TFrame', background='white')

main_frame = ttk.Frame(root)
main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

instruction_label = ttk.Label(main_frame, text="Please choose an option:", font=('Arial', 14, 'bold'))
instruction_label.pack(expand=True, fill=tk.BOTH, pady=5)

browse_button = ttk.Button(main_frame, text="Browse Image", command=browse_file)
browse_button.pack(expand=True, fill=tk.BOTH, pady=10)

webcam_button = ttk.Button(main_frame, text="Open Webcam", command=open_webcam)
webcam_button.pack(expand=True, fill=tk.BOTH, pady=10)

quit_button = ttk.Button(main_frame, text="Quit", command=quit_app)
quit_button.pack(expand=True, fill=tk.BOTH, pady=10)

root.mainloop()