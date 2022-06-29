import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import javax.imageio.ImageIO;

public class NeuralNetwork 
{
   public int neuronsLayer ; //hiddenLayer
   public int epochs ;
   public int pixel;
   public boolean flage =false;
   public double outputLayer; 
   public int[]  predictedLabel ;
   public double[][] weights;
   public double[] weightsOuput;
   public double [] sigmoid ;
   public NeuralNetwork(int neuronsLayer,int pixel)
   {
	   this.neuronsLayer=neuronsLayer;
	   this.pixel=pixel;
   }
   public NeuralNetwork(int neuronsLayer,int epochs,int pixel)
   {
	   this.neuronsLayer=neuronsLayer;
	   this.epochs=epochs;
	   this.pixel=pixel;
	   inputWeights();
	   outputWeights();
   }
   public void inputWeights() //between input and hiddenLayer
   {
	   Random r = new Random();
	   weights = new double[neuronsLayer][pixel];
	   flage=true;
	   for(int i=0 ; i<neuronsLayer ;i++)
       {
       	for (int j = 0; j<pixel; j++) 
           {
       		 double random = -1 + r.nextDouble() * (1 - (-1));
       		 double weight = Math.round(random * 100.0) / 100.0;
       		 weights[i][j]=weight;
           }
       }
   }
   public void outputWeights() //between hiddenLayer and output
   {
	   Random r = new Random();
	   weightsOuput =new double [neuronsLayer];
	   for(int i=0 ; i<neuronsLayer ;i++)
       {
       		 double random = -1 + r.nextDouble() * (1 - (-1));
       		 double weight = Math.round(random * 100.0) / 100.0; 
       		 weightsOuput[i]=weight;
       }
   }
   public void train(int[][] trainingSetFeatures,int[] actualLabels)
   {
	   
	   for(int i=0;i<epochs;i++)
	   {
		   for(int j=0;j<30;j++)
		   {
			   trainOneEx(trainingSetFeatures[j],actualLabels[j]);
		   }
	  }
   }
   public void trainOneEx(int[]example,int label)
   {
	  // double weight;
	   forward(example);
	   backPropagation(example,outputLayer,label,sigmoid);
	   
   }
   public void forward(int[]example)
   {
	   double[]output =new double[neuronsLayer] ;
	   sigmoid = new double[neuronsLayer];
	   outputLayer=0;
	   for(int i=0 ; i<neuronsLayer;i++)
       {
		   output[i]=0;
       		 for(int k=0 ;k<pixel;k++)
       		 {
       	     	 output[i]+=weights[i][k]*example[k];	
       		 }
       }
	   //sigmoid function = 1/1+e^-x
	    //output le hiddenLayer
	   double sigmoidNum =0; 
	   double approximately= 2.718;
	   for(int i=0 ; i<neuronsLayer;i++)
	   {
			   sigmoidNum= Math.pow(approximately,(-output[i]));
			   double sigma=1/(1+sigmoidNum);
			    sigmoid[i] = Math.round(sigma * 100.0) / 100.0;  
	   }
	   double sum = 0 ;
	   for(int i=0 ; i<neuronsLayer;i++)
       {
       	    sum+=weightsOuput[i]*sigmoid[i];	
       }
	   
	   //sigmoid function = 1/1+e^-x
	     //between hiddenlayer and outputlayer 
	   sigmoidNum= Math.pow(approximately,(-sum));
	   double sigma=1/(1+sigmoidNum);
	   outputLayer=Math.round(sigma * 100.0) / 100.0; 
   }
   public void backPropagation(int[]example ,double fOutput ,int label,double[] hiddenLayer) 
   {
	   //error y1=y1(1-y1)(target-y1)
	   double error;
	   double n=0.5;
	   double[] errorHiddenLayer= new double[neuronsLayer];
	   error = fOutput*(1-fOutput)*(label-fOutput);
	   //errorHiddenLayer = hiddenLayer[i](1- hiddenLayer[i])(error*
	   for(int i=0;i<neuronsLayer;i++)
	   {
		   errorHiddenLayer[i]=hiddenLayer[i]*(1-hiddenLayer[i])*(error*weightsOuput[i]);
	   }
	 //weightofOutput ely mwgod+[n*error*sigmoid]
	   for(int i=0;i<weightsOuput.length;i++) 
	   {
		   double w= weightsOuput[i]+(error*n*hiddenLayer[i]);
		   double weight = Math.round(w * 100.0) / 100.0; 
		   weightsOuput[i]=weight;
	   }
	 //weightofInput = weight[i]+(n*errorHiddenLayer[i]*trainingSetFeatures[i])
	   for(int i=0 ; i<neuronsLayer ;i++)
       {
       	for (int j = 0; j<pixel; j++) 
           {
       		 
       		   double w = weights[i][j]+(n* errorHiddenLayer[i]*example[j]);
       		   double weight = Math.round(w * 100.0) / 100.0; 
       		   weights[i][j]=weight;
           }
       }
	   
   }
   //testingSetFeatures=new int [10][pixel];
   public int[] predict(int [][]testingSetFeatures)
   {
	   predictedLabel =new int[10];
	   for(int i=0;i<10;i++)
	   {
		   forward(testingSetFeatures[i]);
		   if(0.5<outputLayer) 
		   {
			   predictedLabel[i]=0;
		   }
		   else
		   {
			     predictedLabel[i]=1;
		   }
			  
	   }
	 return predictedLabel;  
   }
   public double calculateAccuracy(int[]predictedLabels,int[] testingSetLabels)
   {
	   double accuracy = 0;
	   double counter =0;
	   for(int i=0;i<predictedLabels.length;i++)
	   {
		   if(predictedLabels[i]==testingSetLabels[i])
		   {
			   counter++;
		   }
	   }
	   accuracy = counter/10.0*100;
	   
	   return accuracy;
   }
   public void save(String filename) throws IOException
   {
	    FileWriter writer = new FileWriter(filename);
	    PrintWriter pw = new PrintWriter(writer);
	    for(int i=0 ;i<neuronsLayer;i++)
	    {
	    	for (int j = 0; j<pixel; j++) 
	         {
	    		 pw.println(weights[i][j]);
	         }
	    }
	    for(int i=0 ;i<neuronsLayer;i++)
	    {
	    	pw.println(weightsOuput[i]);
	    }
	    pw.close();
   }
   public void load(String filename) throws IOException
   {
	   File file = new File(filename);
	   BufferedReader br= new BufferedReader(new FileReader(file));
	   String line;
	   weights=new double[neuronsLayer][pixel];
	   weightsOuput=new double[neuronsLayer];
	   int lineCount = (neuronsLayer*pixel)+neuronsLayer;
	   int n = (neuronsLayer*pixel);
	   String[] lineFiles = new String[lineCount];
	   int i=0;
	   while ((line = br.readLine()) != null)
	    {
	        lineFiles[i]=line;
	         i++;
        }
	   for(i=neuronsLayer-1;i>=0;i--)
	   {
		   for(int j=n-1,k=pixel-1;j>=(n-pixel)&&k>=0;j--,k--)
		   {
			   weights[i][k]=Double.parseDouble(lineFiles[j]);
		   }
		    n-=pixel;
	   }
	   for(int k=lineCount-1,j=neuronsLayer-1;k>=n&&j>=0;j--,k--)
	   {
		   weightsOuput[j]=Double.parseDouble(lineFiles[k]);
	   }
   }
   public int predict(int[]sampleImgFeatures)
   {
	       int out;
		   forward(sampleImgFeatures);
		   if(0.5<outputLayer) 
		   {
			   out=0;
		   }
		   else
		   {
			   out=1;
		   }
			  
	return out;
	   
   }
}
