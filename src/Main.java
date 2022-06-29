import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class ImageData {
    int[] pixels;
    int label;
    ImageData(){
    	
    }
    ImageData(int imgHeight, int imgWidth) { pixels = new int[imgHeight*imgWidth]; }
    public void setPixels(int[] pixels) { this.pixels = pixels; }
    public void setLabel(int lbl) { label = lbl; }
}

public class Main {
    public static void main(String[] args) throws IOException {
        //Load Data
        File[] images= new File("Cats & Dogs Sample Dataset").listFiles();
        ImageData[] data = new ImageData[images.length];
        for (int i = 0; i < images.length; i++) 
        {
        	data[i]=new ImageData();
            data[i].setPixels(ImageHandler.ImageToIntArray(images[i]));
            data[i].setLabel(images[i].getName().contains("cat")? 0 : 1);
        }
        //Shuffle mixing 
        List<ImageData> tempData = Arrays.asList(data);
        Collections.shuffle(tempData);
        tempData.toArray(data);

        //Split the data into training (75%) and testing (25%) sets pic=40
        int pixel = tempData.get(0).pixels.length; //1600

        int[][] trainingSetFeatures = new int [30][pixel],testingSetFeatures=new int [10][pixel]; //inputs x1 x2 and so on 
        int[] trainingSetLabels = new int [30], testingSetLabels=new int [10]; //output y1 cat or dog
        
        //----------training--------------(75%)=30
        for(int i=0 ; i<30 ;i++)
        {
        	for (int j = 0; j<pixel; j++) 
            {
        		trainingSetFeatures[i][j]=tempData.get(i).pixels[j];
            }
        	trainingSetLabels[i] =tempData.get(i).label;
        	
        }
        
      //----------Testing--------------(25%)=10
        for(int i=0 ; i<10 ;i++)
        {
        	for (int j = 0; j<pixel; j++) 
            {
        		testingSetFeatures[i][j]=tempData.get(i).pixels[j];
            }
        	testingSetLabels[i] =tempData.get(i).label;
        }
        
        //Create the NN
        NeuralNetwork nn = new NeuralNetwork(3,100,pixel);
        //Train the NN
        
        nn.train(trainingSetFeatures, trainingSetLabels);

        //Test the model
        int[] predictedLabels = nn.predict(testingSetFeatures);
        double accuracy = nn.calculateAccuracy(predictedLabels, testingSetLabels);
        System.out.println("accuracy: "+accuracy);
        //Save the model (final weights)
        nn.save("model.txt");

        //Load the model and use it on an image
        NeuralNetwork nn2 =new NeuralNetwork(3,pixel);
        nn2.load("model.txt");
        int[] sampleImgFeatures = ImageHandler.ImageToIntArray(new File("sample.jpg"));
        int samplePrediction = nn2.predict(sampleImgFeatures);
        ImageHandler.showImage("sample.jpg");
        if(samplePrediction==0)
        	System.out.println("ana cat");
        else System.out.println("ana klb");
        //Print "Cat" or "Dog"
        /*
            ...
         */
    }
}