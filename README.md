# IBY1.0_Object_Detection
This project is a WPF application that detects objects from webcam using a pre-trained YOLOv4 ONNX model.                
               
Steps to setup the project -                
  1. Create a new WPF App (.NET framework = 4.7.2) Project in Microsoft Visual Studio 2019.                               
   _**Note:** Preferably name the project as "IBY1.0", if not, change the namespace names and other references to your project name accordingly._                        
  2. Clone this repository into your project folder. Extract all the contents of the folder IBY1.0 in the repository to your Project folder.                
  3. Go to Project->Manage NuGet Packages and install the following packages -                
       - Microsoft.ML v1.5.5               
       - Microsoft.ML.ImageAnalytics v1.6.0                
       - Microsoft.ML.OnnxTransformer v1.5.5               
       - Microsoft.ML.OnnxRuntime v1.8.1               
       - EmguV v3.1.0.1               
  4. Next, Download the pre-trained ONNX model from the link given below, and paste it in ./assets/Model Folder. Make sure the name of the .onnx file is "Yolov4_model.onnx".                
        Link for .onnx file - https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx               
  5. The project is ready to use.                

  In case the following setup does not work, the entire project will all the libraries and the .onnx file is available in the google drive here -               
        Link for project on google drive - https://drive.google.com/drive/folders/1FMvYHsKBtnClpn2f6Rtlt77XhcX6oOl8?usp=sharing               
        
       
