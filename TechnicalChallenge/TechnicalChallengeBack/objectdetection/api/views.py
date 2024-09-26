import os
import cv2
import platform
from . import objectCounter 
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from ultralytics import YOLO

class VideoUploadView(APIView):
    parserClasses = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        videoFile = request.FILES.get('video')

        if not videoFile:
            return Response({'error': 'Nenhum vídeo foi enviado.'}, status = status.HTTP_400_BAD_REQUEST)

        rootDirectory = getRootDirectory()
        videoPath = getFileName(rootDirectory, 'uploaded_videos', videoFile.name)
        processedVideoPath = getFileName(rootDirectory, 'processed_videos', f'processed_{videoFile.name}')

        createDirectory(videoPath)
        createDirectory(processedVideoPath)

        with open(videoPath, 'wb+') as destination:
            for chunk in videoFile.chunks():
                destination.write(chunk)

        peoplesCount, objectsCount = proccessVideo(videoPath, processedVideoPath)

        return Response({
            'message': 'Vídeo processado com sucesso!',
            'processedVideoPath': processedVideoPath,
            'peoplesCount': peoplesCount,
            'objectsCount': objectsCount
        }, status = status.HTTP_200_OK)
    



    
def getRootDirectory():
    system = platform.system()
    if system == 'Windows':
        return 'C:\\'
    else:
        return '/'





def getFileName(directory: str, folder: str, fileName: str):
    return os.path.join(directory, folder, fileName)





def createDirectory(directoryName: str):
    os.makedirs(os.path.dirname(directoryName), exist_ok=True)





def proccessVideo(videoPath: str, outputPath: str):
    model = YOLO('yolov8n.pt')
    capture = cv2.VideoCapture(videoPath)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))
    peoplesCount = 0
    objectsCount = 0

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label = model.names[int(box.cls[0])]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                if label == 'person':
                    peoplesCount += 1
                else:
                    objectsCount += 1                

        cv2.putText(frame, 'Pessoas: ' + str(peoplesCount), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, 'Pessoas: ' + str(peoplesCount), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, 'Objetos: ' + str(objectsCount), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, 'Objetos: ' + str(objectsCount), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        peoplesCount = 0
        objectsCount = 0

        out.write(frame)

    capture.release()
    out.release()

    return peoplesCount, objectsCount