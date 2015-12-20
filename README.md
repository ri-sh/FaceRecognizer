# Face Recognition using Eigenfaces/  Fisherfaces


### Detecting Faces  
Perform face detection on the input image, using the given Haar Cascade.
 Return a rectangle for the detected region in the given image


```C++ 
CvRect detectFaceInImage(IplImage *inputImg, CvHaarClassifierCascade* cascade)
{
	// Smallest face size.
	CvSize minFeatureSize = cvSize(20, 20);
	// Only search for 1 face.
	int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH;
	// How detailed should the search be.
	float search_scale_factor = 1.1f;
	IplImage *detectImg;
	IplImage *greyImg = 0;
	CvMemStorage* storage;
	CvRect rc;
	double t;
	CvSeq* rects;
	CvSize size;
	int i, ms, nFaces;

	storage = cvCreateMemStorage(0);
	cvClearMemStorage( storage );


	// If the image is color, use a greyscale copy of the image.
	detectImg = (IplImage*)inputImg;
	if (inputImg->nChannels > 1) {
		size = cvSize(inputImg->width, inputImg->height);
		greyImg = cvCreateImage(size, IPL_DEPTH_8U, 1 );
		cvCvtColor( inputImg, greyImg, CV_BGR2GRAY );
		detectImg = greyImg;	// Use the greyscale image.
	}

	// Detect all the faces in the greyscale image.
	t = (double)cvGetTickCount();
	rects = cvHaarDetectObjects( detectImg, cascade, storage,
			search_scale_factor, 3, flags, minFeatureSize);
	t = (double)cvGetTickCount() - t;
	ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
	nFaces = rects->total;
	printf("Face Detection took %d ms and found %d objects\n", ms, nFaces);

	// Get the first detected face (the biggest).
	if (nFaces > 0)
		rc = *(CvRect*)cvGetSeqElem( rects, 0 );
	else
		rc = cvRect(-1,-1,-1,-1);	// Couldn't find the face.

	if (greyImg)
		cvReleaseImage( &greyImg );
	cvReleaseMemStorage( &storage );
	//cvReleaseHaarClassifierCascade( &cascade );

	return rc;	// Return the biggest face found, or (-1,-1,-1,-1).
}
```
using the haar cascade frontal face  classifiers in the above functions 
For frontal face detection, you can chose one of these Haar Cascade Classifiers that come with OpenCV (in the "data\haarcascades\" folder):
* "haarcascade_frontalface_default.xml"
* "haarcascade_frontalface_alt.xml"
* "haarcascade_frontalface_alt2.xml"
* "haarcascade_frontalface_alt_tree.xml"

```{
  char *faceCascadeFilename = "haarcascade_frontalface_alt.xml";
// Load the HaarCascade classifier for face detection.
CvHaarClassifierCascade* faceCascade;
faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);
if( !faceCascade ) {
	printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
	exit(1);
}

// Grab the next frame from the camera.
IplImage *inputImg = cvQueryFrame(camera);

// Perform face detection on the input image, using the given Haar classifier
CvRect faceRect = detectFaceInImage(inputImg, faceCascade);

// Make sure a valid face was detected.
if (faceRect.width > 0) {
	printf("Detected a face at (%d,%d)!\n", faceRect.x, faceRect.y);
}

.... Use 'faceRect' and 'inputImg' ....

// Free the Face Detector resources when the program is finished
cvReleaseHaarClassifierCascade( &cascade );

```


###  preprocessing  facial images for Face Recognition



Face ----> Greyscale ----> Equalized(histogram equalized face )




```code
//Either convert the image to greyscale, or use the existing greyscale image.

IplImage *imageGrey;
if (imageSrc->nChannels == 3) {
	imageGrey = cvCreateImage( cvGetSize(imageSrc), IPL_DEPTH_8U, 1 );
	// Convert from RGB (actually it is BGR) to Greyscale.
	cvCvtColor( imageSrc, imageGrey, CV_BGR2GRAY );
}
else {
	// Just use the input image, since it is already Greyscale.
	imageGrey = imageSrc;
}

// Resize the image to be a consistent size, even if the aspect ratio changes.
IplImage *imageProcessed;
imageProcessed = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
// Make the image a fixed size.
// CV_INTER_CUBIC or CV_INTER_LINEAR is good for enlarging, and
// CV_INTER_AREA is good for shrinking / decimation, but bad at enlarging.
cvResize(imageGrey, imageProcessed, CV_INTER_LINEAR);

// Give the image a standard brightness and contrast.
cvEqualizeHist(imageProcessed, imageProcessed);


if (imageGrey)
	cvReleaseImage(&imageGrey);
if (imageProcessed)
	cvReleaseImage(&imageProcessed);

```

### Using the  cmd based Face Recognizer System



###### use the realtime webcam FaceRec system
on cmd type Display or double click Display.exe file to run the system 

* In the console window, hit the 'n' key on your keyboard when a person is ready for training. This will add a new person to the facerec database. Type in the person's name (without any spaces) and hit Enter.
* it will begin to automatically store all the processed frontal faces that it sees. Get a person to move their head around a bit until it has stored about 20 faces of them. (The facial images are stored as PGM files in the "data" folder, and their names are appended to the text file "train.txt").
* Get the person in front of the camera to move around a little and move their face a little, so that there will be some variance in the training images.
* Then when you have enough detected faces for that person, ideally more than 30 for each person, hit the 't' key in the console window to begin training on the images that were just collected. It will then pause for about 5-30 seconds (depending on how many faces and people are in the database), and finally continue once it has retrained with the extra person. The database file is called "facedata.xml".
* It should print the person's name in the console whenever it recognizes them.

Repeat this again from step 1 whenever you want to add a new person, even after you have shutdown the program.
If you hit the Escape key in the console (or GUI) then it will shutdown safely. You can then run the program again later and it should already be trained to recognize anyone that was added.

######   use the command-line mode for offline facerec here are the instructions: 
* First, you need some face images. You can find many face databases at the Essex page http://peipa.essex.ac.uk/benchmark/databases/index.html. I used "ORL / AT&T: The Database of Faces": Cambridge_FaceDB.zip (3.7MB).
* List the training and test face images you want to use into text files. If you downloaded the ORL database, you can use my sample text files: facerecExample_ORL.zip (1kB). To use these input files exactly as provided, unzip the facerecExample_ORL.zip file in your folder with Display.exe and then unzip the ORL face database there (eg: 'Cambridge_DB\s1\1.pgm').
* To run the learning phase of eigenface, enter in the command prompt: 
  	Display train lower2.txt 
     That will create a database file "facedata.xml" with just 2 faces each from 10       people (a total of 20 faces). It will also generate "out_averageImage.bmp" and      "out_eigenfaces.bmp" for you to look at.
*  To run the recognition phase, enter: 
  	Display test upper6.txt 
(That will test the database with 6 faces each from 10 people (a total of 60 faces). 
It should give a surprisingly high recognition rate of 95% correct, from just 2 photos of each person!)





![Eigen Faces](https://raw.githubusercontent.com/rishabhsixfeet/FaceRecognizer/master/out_eigenfaces.bmp)