import au.com.bytecode.opencsv.CSVWriter;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.FacemarkKazemi;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.Math.*;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_core.rotate;
import static org.bytedeco.opencv.global.opencv_face.drawFacemarks;

public class drowsinessDetector {
    private static final Logger log = LoggerFactory.getLogger(drowsinessDetector.class);

    private static final int cameraNum = 0;
    private static final String applyOn = "cam"; // video Or camqq
    private static final String winName = "Object Detection";
    private static final String dataFolder = "dataset";
    private static final DecimalFormat df = new DecimalFormat("##.##");

    private static List<String> videoPaths = new ArrayList<>();
    private static String cameraPos = "front"; // front or back
    private static String type, label, featuresFile, labelsFile, format;
    private static Thread thread;

    public static void main(String[] args) throws Exception {

        if (applyOn.equals("video")) {
            showFiles(new File(dataFolder).listFiles());

            int counter = 0;
            for (String videoPath: videoPaths) {
                System.out.println("Currently Processing Video: " + videoPath);
                Matcher m = Pattern.compile(".*(\\d{2}).(\\d+).(.{3})$").matcher(videoPath);
                if (m.find()) {
                    // Group 1 = folder (people)
                    // Group 2 = status
                    type = m.group(1);
                    label = m.group(2);
                    featuresFile = "output/features/" + counter + ".csv";
                    labelsFile = "output/labels/" + counter + ".csv";
                    format = m.group(3);
                }
                getRecord(videoPath);
                counter++;
            }
        } else {
            getRecord();
        }
    }

    public static String[] getFacemarks(Point2fVector v, Mat rawImage) throws Exception {
        double[][] left_eye = getXY(v, 36, 41, "Left eye");
        double[][] right_eye = getXY(v, 42, 47, "Right eye");
        double[][] mouth = getXY(v, 60, 67, "Mouth");

        double EAR_left = (
                getEuclidianDist(left_eye[1], left_eye[5])
                        + getEuclidianDist(left_eye[2], left_eye[4])
        )
                / (2 * getEuclidianDist(left_eye[0], left_eye[3]));

        double EAR_right = (
                getEuclidianDist(right_eye[1], right_eye[5])
                        + getEuclidianDist(right_eye[2], right_eye[4])
        )
                / (2 * getEuclidianDist(right_eye[0], right_eye[3]));

        double MAR = getEuclidianDist(mouth[6], mouth[2])
                / getEuclidianDist(mouth[0], mouth[4]);

        double circularity_left = getCircularity(left_eye);
        double circularity_right = getCircularity(right_eye);

        double EAR = (EAR_left + EAR_right) / 2;
        double circularity = (circularity_left + circularity_right) / 2;
        double MOE = MAR / EAR;

        drawFacemarks(rawImage, v, Scalar.YELLOW);

        /*
        System.out.println("EAR: " + EAR);
        System.out.println("EAR (Left): " + EAR_left);
        System.out.println("EAR (Right): " + EAR_right);
        System.out.println("Circularity: " + circularity);
        System.out.println("Circularity (Left): " + circularity_left);
        System.out.println("Circularity (Right): " + circularity_right);
        System.out.println("MOE: " + MOE);
        System.out.println("MAR: " + MAR);
        */

        String sEAR = Double.toString(EAR);
        String sCir = Double.toString(circularity);
        String sMOE = Double.toString(MOE);
        String sMAR = Double.toString(MAR);

        String[] requiredLandmarks = {sEAR, sCir, sMOE, sMAR};

        return requiredLandmarks;
    }

    public static void getRecord() throws Exception {
        FrameGrabber grabber = getCamGrabber();
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        // This is because the size is transposed when receives video
        int w = grabber.getImageHeight();
        int h = grabber.getImageWidth();

        CanvasFrame canvas = new CanvasFrame(winName);
        canvas.setCanvasSize(w, h);

        // https://github.com/bytedeco/javacv/blob/master/samples/KazemiFacemarkExample.java
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt2.xml");
        FacemarkKazemi facemark = FacemarkKazemi.create();
        facemark.loadModel("face_landmark_model.dat");

        log.info("Start running video");

        while ((grabber.grab()) != null) {
            Frame frame = getFrame(grabber);
            Mat rawImage = getRawImage(frame, converter);

//            Mat resizeImage = rawImage;
//            resize(rawImage, resizeImage, new Size(tinyyolowidth, tinyyoloheight));

            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(rawImage, faces);

            Point2fVectorVector landmarks = new Point2fVectorVector();
            boolean success = false;

            try {
                success = facemark.fit(rawImage, faces, landmarks);
            } catch (Exception e) { }

            if (success) {
               /* As we only want 1 faces
               for (long i = 0; i < landmarks.size(); i++) {
                   Point2fVector v = landmarks.get(i);
                   double[][] left_eye = getXY(v, 36, 41, "Left eye");
                   double[][] right_eye = getXY(v, 42, 47, "Right eye");
                   double[][] mouth = getXY(v, 48, 67, "Mouth");

                   drawFacemarks(rawImage, v, Scalar.YELLOW);
               }*/

                Point2fVector v = landmarks.get(0);
                String[] landmark = getFacemarks(v, rawImage);
            }
            canvas.showImage(converter.convert(rawImage));

            KeyEvent t = canvas.waitKey(33);
            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
        canvas.dispose();
    }

    public static void getRecord(String videoPath) throws Exception {

        FFmpegFrameGrabber grabber = getVideoGrabber(videoPath, format);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        // This is because the size is transposed when receives video
        int w = grabber.getImageHeight();
        int h = grabber.getImageWidth();

        CanvasFrame canvas = new CanvasFrame(winName);
        canvas.setCanvasSize(w, h);

        // https://github.com/bytedeco/javacv/blob/master/samples/KazemiFacemarkExample.java
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt2.xml");
        FacemarkKazemi facemark = FacemarkKazemi.create();
        facemark.loadModel("face_landmark_model.dat");

        log.info("Start running video");
        CSVWriter writer = new CSVWriter(new FileWriter(featuresFile));
        CSVWriter writerLabel = new CSVWriter(new FileWriter(labelsFile));

        String[] labels = {label};
        writerLabel.writeNext(labels);
        writerLabel.close();

        int frameNum = 0;

        while ((grabber.grab()) != null) {
            Frame frame = getFrame(grabber);
            Mat rawImage = getRawImage(frame, converter);

            RectVector faces = new RectVector();
            faceDetector.detectMultiScale(rawImage, faces);

            if (frameNum < 5400) {
                frameNum++;
                continue;
            }

            Point2fVectorVector landmarks = new Point2fVectorVector();
            boolean success = false;

            try {
                success = facemark.fit(rawImage, faces, landmarks);
            } catch (Exception e) { }

            if (success) {
                Point2fVector v = landmarks.get(0);
                String[] landmark = getFacemarks(v, rawImage);
                writer.writeNext(landmark);
            }

            canvas.showImage(converter.convert(rawImage));

            KeyEvent t = canvas.waitKey(33);
            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }

            if (frameNum == 12600) {
                break;
            }

            frameNum++;
        }
        canvas.dispose();
        writer.close();
    }

    public static void showFiles(File[] files) {
        for (File file: files) {
            if (file.isDirectory()) {
                showFiles(file.listFiles());
            } else {
                videoPaths.add(file.getAbsolutePath());
            }
        }
    }

    public static double getCircularity(double[][] obj) {
        double area = pow((getEuclidianDist(obj[1], obj[4]) / 2), 2) * PI;
        double perimeter = getEuclidianDist(obj[0], obj[1]) + getEuclidianDist(obj[1], obj[2])
                + getEuclidianDist(obj[2], obj[3]) + getEuclidianDist(obj[3], obj[4])
                + getEuclidianDist(obj[4], obj[5]) + getEuclidianDist(obj[5], obj[0]);
        double circularity = (4 * PI * area) / pow(perimeter, 2);

        return circularity;
    }

    public static double[][] getXY(Point2fVector point, int loc1, int loc2, String info) {
        int range = loc2 - loc1 + 1;
        double[][] xy = new double[range][2];
        for (int i=0; i < range; i++) {
            Point2f newPoint = point.get(loc1 + i);
            xy[i][0] = newPoint.x();
            xy[i][1] = newPoint.y();
        }
//        print_info(info, xy);
        return xy;
    }

    public static double getEuclidianDist(double[] point1, double[] point2) {
        double dist_x = point2[0] - point1[0];
        double dist_y = point2[1] - point1[1];
        return abs(sqrt(pow(dist_x, 2) + pow(dist_y, 2)));
    }

    public static void print_info(String info, double[][] objs) {
        System.out.printf("Print %s:\n", info);
        for (double[] obj: objs) {
            System.out.println(obj[0] + ", " + obj[1]);
        }
    }

    public static FrameGrabber getCamGrabber() throws Exception {
        if (!cameraPos.equals("front") && !cameraPos.equals("back")) {
            throw new Exception("Unknown argument for camera position. Choose between front and back.");
        }

        FrameGrabber grabber = FrameGrabber.createDefault(cameraNum);
        return grabber;
    }

    public static FFmpegFrameGrabber getVideoGrabber(String path, String format) throws Exception {
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(path);
        grabber.setFormat(format);
        grabber.setFrameRate(24);
        grabber.setFrameNumber(100);
        return grabber;
    }

    public static Frame getFrame(FFmpegFrameGrabber grabber) throws Exception {
        return grabber.grabImage();
    }

    public static Frame getFrame(FrameGrabber grabber) throws Exception {
        return grabber.grab();
    }

    public static Mat getRawImage(Frame frame, OpenCVFrameConverter.ToMat converter) throws Exception {
        Mat inputImage = converter.convert(frame);

        // Flip the camera if opening front camera
        if (cameraPos.equals("front")) {
            flip(inputImage, inputImage, 1);
            flip(inputImage, inputImage, 0);
        }

        if (applyOn.equals("video")) {
            if (type.equals("01")) {
                if (label.equals("0")) {
                    rotate(inputImage, inputImage, 2);
                } else {
                    rotate(inputImage, inputImage, 1);
                }
            } else {
                rotate(inputImage, inputImage, 1);
            }
        } else {
            rotate(inputImage, inputImage, 1);
        }
        return inputImage;
    }
}


