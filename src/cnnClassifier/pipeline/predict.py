import numpy as np
from tensorflow.keras.models import load_model          #load a pre-trained neural network model saved in the Hierarchical Data Format (HDF5) file format.
from tensorflow.keras.preprocessing import image        #image module provides tools for working with image data, such as loading and preprocessing images.
import os                                               #os module provides a way to interact with the operating system, including functions for file and directory manipulation.



class PredictionPipeline:           # declaring class PredictionPipeline
    def __init__(self,filename):    #   __init__  is constructor. It takes one argument, filename, and 
        self.filename =filename     # sets it as an instance variable (self.filename) to store the name of the image file.


    
    def predict(self):      #predict method is used to make predictions using the loaded model.              
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename  # image vraibale created
        #test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.load_img(imagename, target_size=(256, 256))  # image.load_img from TensorFlow's Keras preprocessing module resizes the   image to 256x256 pixels.
        
        test_image = image.img_to_array(test_image) #The loaded image is converted into a NumPy array using image.img_to_array. This step is necessary to prepare the image data for input to the neural network.
        test_image = np.expand_dims(test_image, axis=0) #np.expand_dims function is used to add an extra dimension to the NumPy array to match the expected input shape 
        # of the model. This is because the model expects a batch of images as input, even if there's only one image in the batch
        predictions = model.predict(test_image) # model is used to make predictions on the preprocessed image 

        class_names = ["Early blight","Late blight" ,"Healthy"]
        predicted_class = class_names[np.argmax(predictions[0])] #The np.argmax function is used to find the index of the class with the highest predicted probability 
        #predictions[0] is used to extract the predictions for the first (and possibly only) image in the batch.
        confidence = round(100 * (np.max(predictions[0])), 2) # arg max will find the index where as max will find the max probability, then multiply by 100 to get the percentage.

        #return [{ "image": predicted_class, "Confidence Level": f"{confidence }%" }]
        # Add HTML tags for bold text
        return [{
            "image": f"<span style='font-size: 12pt; color: blue;'><strong>{predicted_class}</strong></span>",
            "Confidence Level": f"<span style='font-size: 12pt; color: green;'><strong>{confidence}%</strong></span>"
        }]
         



        
        
        
    


        

            
            
        