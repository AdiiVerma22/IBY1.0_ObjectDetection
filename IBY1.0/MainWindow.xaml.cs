using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media.Imaging;
using Emgu.CV;
using System.Windows;
using System.Windows.Threading;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Drawing.Drawing2D;
using Microsoft.ML;
using System.IO;
using IBY1._0.DataStructures;
using Microsoft.ML.Transforms.Onnx;
using Microsoft.ML.Data;

namespace IBY1._0
{
   
    public partial class MainWindow : Window
    {

        private Capture capture;
        DispatcherTimer timer;

        static string assetsRelativePath = @"../../assets";
        static string assetsPath = GetAbsolutePath(assetsRelativePath);
        static string modelPath = Path.Combine(assetsPath, "Model", "Yolov4_model.onnx");

        static MLContext mlContext = new MLContext();
        static EstimatorChain<OnnxTransformer> pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "input_1:0", imageWidth: 416, imageHeight: 416)
       .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1:0", scaleImage: 1f / 255f, interleavePixelColors: true))
       .Append(mlContext.Transforms.ApplyOnnxModel(
           shapeDictionary: new Dictionary<string, int[]>()
           {
                        { "input_1:0", new[] { 1, 416, 416, 3 } },
                        { "Identity:0", new[] { 1, 52, 52, 3, 85 } },
                        { "Identity_1:0", new[] { 1, 26, 26, 3, 85 } },
                        { "Identity_2:0", new[] { 1, 13, 13, 3, 85 } },
           },
           inputColumnNames: new[]
           {
                        "input_1:0"
           },
           outputColumnNames: new[]
           {
                        "Identity:0",
                        "Identity_1:0",
                        "Identity_2:0"
           },
           modelFile: modelPath));
        // recursionLimit: 100

        static TransformerChain<OnnxTransformer> model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV4BitmapData>()));

        // Create prediction engine
        static PredictionEngine<YoloV4BitmapData, YoloV4Prediction> predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV4BitmapData, YoloV4Prediction>(model);


        public MainWindow()
        {
            InitializeComponent();
        }

        public static Bitmap Process_Image( Bitmap image)
        {

            try
            {

                var predict = predictionEngine.Predict(new YoloV4BitmapData() { Image = image });
                var results = predict.GetResults( 0.3f, 0.7f);

                using (var g = Graphics.FromImage(image)){
                    
                    g.CompositingQuality = CompositingQuality.HighQuality;
                    g.SmoothingMode = SmoothingMode.HighQuality;
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    foreach (var res in results)
                    {
                        var x1 = res.BBox[0];
                        var y1 = res.BBox[1];
                        var x2 = res.BBox[2];
                        var y2 = res.BBox[3];
                       
                        g.DrawRectangle(new Pen(Color.Red,3), x1, y1, x2 - x1, y2 - y1);
                        string text = res.Label + " " + res.Confidence.ToString("0.00");
                        SizeF size = g.MeasureString(text, new Font("Arial", 12));                      
                        g.FillRectangle(Brushes.Red,(int)x1,(int)(y1),(int)size.Width,(int)size.Height);
                        g.DrawString(text, new Font("Arial", 12), Brushes.Black, new PointF(x1, y1));
                        
                    }
                }
             //   GC.Collect();
                return image; 
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            return image;
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(MainWindow).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            capture = new Capture();
            timer = new DispatcherTimer();
            timer.Tick += new EventHandler(timer_Tick);
            timer.Interval = new TimeSpan(0, 0, 0, 0, 100);
            timer.Start();
        }

        
        void timer_Tick(object sender, EventArgs e)
        {
            pic1.Source = ToBitmapSource(Process_Image( capture.QueryFrame().Bitmap)); 
        }

        [DllImport("gdi32")]
        private static extern int DeleteObject(IntPtr o);

        public static BitmapSource ToBitmapSource(Bitmap image)
        {
                IntPtr ptr2 = image.GetHbitmap(); //obtain the Hbitmap  
                BitmapSource bs = System.Windows.Interop
                  .Imaging.CreateBitmapSourceFromHBitmap(
                  ptr2,
                  IntPtr.Zero,
                  Int32Rect.Empty,
                  System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());
                DeleteObject(ptr2); 
                return bs;
        }
    }
}
