##########
# Authors: 
# Marcela Fuentes  A01748161
# René García      A01654359
##########

from string import hexdigits
from unittest import skip
from webbrowser import get
import cv2
import time
import numpy as np
import maginner as mg
import getopt
import sys 
import matplotlib.pyplot as plt     

def YoloVideo(input_file, output_file): 
    cap = cv2.VideoCapture(input_file)

    # get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # create the `VideoWriter()` object
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    ## YOLO 
    with open('files/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    yolo_model = cv2.dnn.readNetFromDarknet('files/yolov3.cfg', 'files/yolov3.weights')
    
    ln = yolo_model.getLayerNames()
    ln = [ln[i-1] for i in yolo_model.getUnconnectedOutLayers()]

    # detect objects in each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = frame

            ## TENSOR ##
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            # create blob from image
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), crop=False)

            # create blob from image
            yolo_model.setInput(blob)
            # forward pass through the mobile_net_model to carry out the detection
            outputs = yolo_model.forward(ln)

            boxes = []
            confidences = []
            classIDs = []

            image_copy = np.copy(image)
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.2: 
                        
                        box = detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x,y,int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


            if len(idxs) > 0:
                for i in idxs.flatten(): 
                    (x,y) = (boxes[i][0], boxes[i][1])
                    (w,h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image_copy, (x,y), (x+w, y+h), color, 2)
                    text = f"{class_names[classIDs[i]]}"
                    cv2.putText(image_copy, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,2) 

            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
            
            ######################        
            out.write(image_copy)


            #############
            # cv2.imshow('image', image)
            # out.write(image_copy)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

###########################################################

def TensorVideo(input_file, output_file):
    cap = cv2.VideoCapture(input_file)

    # get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # create the `VideoWriter()` object
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    
    ## TENSOR OPEN 
    with open('files/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    mobile_net_model = cv2.dnn.readNet(model='files/frozen_inference_graph.pb',
                            config='files/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                            framework='TensorFlow')
    
    ret, frame = cap.read()
    image = frame
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image_height, image_width, _ = image.shape
    # create blob from image
    #blob = cv2.dnn.blobFromImage(image=image, size=(300, 300))
    # create blob from image
    #mobile_net_model.setInput(blob)
    # forward pass through the mobile_net_model to carry out the detection
    #output = mobile_net_model.forward()

    # detect objects in each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            
            image = frame
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape
            blob = cv2.dnn.blobFromImage(image=image, size=(300, 300))
            # create blob from image
            mobile_net_model.setInput(blob)
            # forward pass through the mobile_net_model to carry out the detection
            output = mobile_net_model.forward()

            # loop over each of the detection
            for detection in output[0, 0, :, :]:
                # extract the confidence of the detection
                confidence = detection[2]
                # draw bounding boxes only if the detection confidence is above...
                # ... a certain threshold, else skip
                if confidence > .4:
                    # get the class id
                    class_id = detection[1]
                    # map the class id to the class
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    # put the FPS text on top of the frame
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
            out.write(image)
            #############
            # cv2.imshow('image', image)
            # out.write(image_copy)
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

####################

def CaffeVideo(input_file, output_file): 
    
    # Open video
    cap = cv2.VideoCapture(input_file)

    # get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # create the `VideoWriter()` object
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # CAFFE open 
    with open('files/classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')
    
    class_names = [name.split(',')[0] for name in image_net_names]

    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the neural network model
    # model = cv2.dnn.readNet(model='files/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel', config='files/test.prototxt', framework='Caffe')
    #model = cv2.dnn.readNetFromCaffe('files/test.prototxt','files/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    model = cv2.dnn.readNet(model='files/VGG.caffemodel',
                        config='files/VGG.prototxt', 
                        framework='Caffe')
    # detect objects in each frame of the video


    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            image = frame

            image_height, image_width, _ = image.shape
            ## CAFFE LOGINC ##

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #blob = blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
            blob = cv2.dnn.blobFromImage(image=image, size=(300,300))


            model.setInput(blob)

            outputs = model.forward()

            # loop over each of the detection
            for detection in outputs[0, 0, :, :]:
                # extract the confidence of the detection
                confidence = detection[2]
                # draw bounding boxes only if the detection confidence is above...
                # ... a certain threshold, else skip
                if confidence > .4:
                    # get the class id
                    class_id = detection[1]
                    # map the class id to the class
                    class_name = class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    image_copy = np.copy(image)
                    # draw a rectangle around each detected object
                    cv2.rectangle(image_copy, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    # put the FPS text on top of the frame
                    cv2.putText(image_copy, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
            out.write(image_copy)
                
            #out.write(image_copy)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

##############

def Help():
    print("""
    Usage:
        python3 detect_vid.py -h
        python3 detect_vid.py -i input_file -o output_file <-c || -t || -y>
    + - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
        -h --help               usage info
        -i --input              input file
        -o ..output             output file

        < Operation modes > 

            -c --caffe          Caffe
            -t --tensorflow     Tensorflow
            -y --YOLO           YOLO """)


def Menu(argv, argc, lw): 

    waerr = "Try running with -h --help for usage information"

    if argc == 0:
        print("[-] Missing flags: %s" %waerr)
        sys.exit(2)
    for x in range(argc):
        if x == 1 or x == 3:
            continue
        elif argv[x] not in lw: 
            print("[-] Unsopported flag ", argv[x], ": ", waerr)
            sys.exit(2)
    if argv[0] in ('-h', '--help'):
        pass
    elif argc != 5:
        print("[-] Wrong number of flags: %s" %waerr)
        sys.exit(2)
    elif argv[0] not in ('-i', '--input') or argv[2] not in ('-o', '--output'):
        print("[-] Missing or unordered input and output files: %s" %waerr)
        sys.exit(2)
    
    try:
        opts, args = getopt.getopt(argv, 'hi:o:cty', ['help',
                                                      'input',
                                                      'output',
                                                      'caffe',
                                                      'tensorflow',
                                                      'yolo'])
        c = 0
        for f in argv:
            if f in ('-t', '--tensorflow', '-c', '--caffe', '-y', '--yolo'): 
                c += 1
        if argv[0] in ('-h', '--help'):
            c = 1 
        if c != 1: 
            print("[-] Wrong number of operation modes: %s" %waerr)
            sys.exit(2)
    except getopt.GetoptError as err:
        print(err, end = ": %s" %waerr)
        sys.exit(2)
    
    input_file = ""
    output_file = ""

    try:
        for opt, arg in opts:
            if opt in ('-h', '--help'): 
                print("HELP")
                Help()
            elif opt in ('-i', '--input'): 
                input_file = arg
                print("Input file: %s" %input_file)
            elif opt in ('-o', '--output'):
                output_file = arg
                print("Output file: %s" %output_file)
            elif opt in ('-c', '--caffe'):
                Banner("Caffe mode")
                CaffeVideo(input_file, output_file)
            elif opt in ('-t', '--tensorflow'): 
                Banner("Tensorflow mode")
                TensorVideo(input_file, output_file)
            elif opt in ('-y', '--yolo'):
                Banner("YOLO mode")
                YoloVideo(input_file, output_file)
            else:
                print("[-] Unsupported flag: ", opt, ": ", waerr)
    except Exception as err:
        print("[-] Something went wrong: ", err)

def Banner(title): 
    mg.maginner(title)
    print("""
    + Marcela Fuentes A01748161
    + René García     A01654359
    """)

def main():
    # Get arguments 
    argv = sys.argv[1:]
    argc = len(argv)

    lw = ['-h', '--help', '-i', '--input', '-o', '--output', '-c', 
          '--caffe', '-t', '--tensorflow', '-y', '--yolo']
    
    Menu(argv, argc, lw)

if __name__ == '__main__':

    main()
