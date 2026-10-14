import cv2 
import numpy as np
import base64
from pathlib import Path
from fastmcp import FastMCP
import joblib

BASE = Path(r'C:\Users\ejjig\OneDrive\Desktop\coal_mcp')



def load_files():
    missing = []
    for fname in ["best_model.pkl", "scaler.pkl", "label_encoder.pkl"]:
        if not (BASE/fname).exists():
            missing.append(fname)
    if missing:
        raise FileNotFoundError(f'missing files:{missing}. server cannot start.')
    
    model = joblib.load(open(BASE/ 'best_model.pkl','rb'))
    scaler = joblib.load(open(BASE/ 'scaler.pkl','rb'))
    le = joblib.load(open(BASE/ 'label_encoder.pkl','rb'))
    
    return model,scaler,le
    
model ,scaler,le = load_files()


# lets create mcp server 
app = FastMCP('Coal-grade_server') 


# preprocessing pipeline
def preprocess(image):
    image = cv2.resize(image,(224,224))
    brightness = np.mean(image)
    if brightness < 50:
        image =  cv2.GaussianBlur(image, (5, 5), 0)
    return image
    
    

# color histograms 
def extract_color_histogram(image, bins=32):
    hist_features = []
    
    for channel in range(3):  
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
        
    return np.array(hist_features)

#decoding image
def decode_image(image_base64:str):
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes,dtype=np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError('could not decode image-invalid base64 or corrupted image')
    return image
    
# tool def
@app.tool()
def grade_coal(image_base64: str) -> str:
    
    """
    Classifies a coal ore image into one of four quality grades
    (grade_a, grade_b, grade_c, reject) using color histogram analysis.
    Accepts a base64 encoded BGR image string as input.
    Returns the predicted grade and confidence score.
    """

    
    #decode
    image = decode_image(image_base64)
            
    #preprocess
    processed_image = preprocess(image)
            
    #extract features
    features = extract_color_histogram(processed_image)
            
    #scaling
    scaled = scaler.transform([features])
            
    #predict
    prediction = model.predict(scaled)[0]
    confidence = model.predict_proba(scaled)[0].max()
            
    # decode
    grade = le.inverse_transform([prediction])[0]
            
    return  f'Grade:{grade} | Confidence: {confidence:.2%}'



# resource
@app.resource("coal://info")
def get_info() -> str:
    return """
COAL GRADE QUALITY CLASSIFICATION SYSTEM

Model Information:
- Algorithm: Logistic Regression
- Features: Color histogram
- Features count: 96
- Histogram bins: 32 per channel
- Channels: B, G, R
- Preprocessing: Image resizing and conditional denoising

Coal Grade Definitions:
- grade_a: Highest quality grade in the classification system
- grade_b: Medium quality grade
- grade_c: Lower quality grade
- reject: Lowest classification / not suitable according to the model

Quality Hierarchy:
grade_a > grade_b > grade_c > reject

Important:
The model predicts grades using visual color-histogram features.
It does not directly measure carbon content, ash content,
chemical composition, or impurity percentage.
"""

# prompt
@app.prompt("classify-with-reasoning")
def classify_with_reasoning() -> str:
    return """
You are a coal quality analysis assistant.

When classifying a coal image:

1. Call the grade_coal() tool.
2. Report the predicted grade.
3. Report the confidence score.
4. Use coal://info to explain what that grade represents.
5. Explain that the model uses visual color-histogram features.
6. Do not claim that the image directly proves carbon content,
   ash content, chemical composition, or impurity percentage.
7. Recommend expert or laboratory verification for actual
   chemical coal quality.

Format:

- Predicted Grade:
- Confidence:
- Quality Level:
- Model Basis:
- Recommendation:
"""


# entrypoint
if __name__ == "__main__":
    app.run()
    














