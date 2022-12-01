import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import File
from fastapi import UploadFile
import torch
import torch.nn as nn
from pydantic import BaseModel
from image_preprocessor import image_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageClassifier(nn.Module):
    def __init__(self, decoder: dict):
        super().__init__()
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        resnet50.fc = torch.nn.Sequential(
            torch.nn.Linear(resnet50.fc.in_features, 13),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim = 1)
        )
        self.layers = resnet50
        self.decoder = decoder

    def forward(self, image):
        predictions = self.layers(image)
        return predictions 

    def predict(self, image):
        '''
        Predicts the category of the image.
        Return the name of the category (i.e. decoded).
        '''
        with torch.no_grad():
            predictions = self.forward(image)
        return self.decoder[int(torch.argmax(predictions, dim=1))]
    
    def predict_categories(self, image):
        '''
        Returns a dictionary of probabilities for each category.
        '''
        with torch.no_grad():
            predictions = self.forward(image)
            predictions = predictions.squeeze() # tensor of length 13
            prob_by_category = {}
            for idx, prob in enumerate(predictions):
                prob_by_category.update({self.decoder[idx]: round(float(prob), 5)})
        return prob_by_category


# Load decoder and image model
try:
    with open('image_decoder.pkl', 'rb') as f:
        decoder = pickle.load(f)
except:
    raise OSError("No decoder found. Check that you have the decoder in the correct location")

try:
    model = ImageClassifier(decoder)
    model.to(device)
    state = torch.load("TransferResnet50_32epochs.pt")
    model.load_state_dict(state['model_state_dict'], strict = False)
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
    msg = "API is up and running!"
    return {"message": msg}
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    '''
    Process the uploaded image and predict its category with our model.
    Returns a dictionary with the predicted category and a list of probabilities for each category.
    '''
    pil_image = Image.open(image.file)
    im_processed = image_process(pil_image).to(device)
    predicted_class = model.predict(im_processed)
    all_probabilities = model.predict_categories(im_processed)
    response = {
    "Category": predicted_class, # Return the category here
    "Probabilities": all_probabilities # Return a list or dict of probabilities here
    }
    return JSONResponse(content=response)

if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8080)