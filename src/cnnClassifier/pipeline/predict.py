import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        #test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.load_img(imagename, target_size=(256, 256))
        
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(test_image)

        class_names = ["Early blight","Late blight" ,"Healthy"]
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)

        #return [{ "image": predicted_class, "Confidence Level": f"{confidence }%" }]
        # Add HTML tags for bold text
        return [{
            "image": f"<span style='font-size: 12pt; color: blue;'><strong>{predicted_class}</strong></span>",
            "Confidence Level": f"<span style='font-size: 12pt; color: green;'><strong>{confidence}%</strong></span>"
        }]
         



        
        
        
    


        

            
            
        