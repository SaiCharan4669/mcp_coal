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
        image =  cv2.fastNlMeansDenoisingColored(image,None,7,10,7,21)
    return image
    
    return denoised

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
    denoised = preprocess(image)
            
    #extract features
    features = extract_color_histogram(denoised)
            
    #scaling
    scaled = scaler.transform([features])
            
    #predict
    prediction = model.predict(scaled)[0]
    confidence = model.predict_proba(scaled)[0].max()
            
    # decode
    grade = le.inverse_transform([prediction])[0]
            
    return  f'Grade:{grade} | Confidence: {confidence:.2%}'


# entrypoint
if __name__ == "__main__":
    app.run()
    

# resource
@app.resource('coal://info')
def get_info()-> str:
    """ Provides Coal grading system information includes grade info,model details,dataset info."""
    return """
COAL GRADE QUALITY CLASSIFICATION SYSTEM

Model info:
-Algorithm : Logistic Regression
- Features: Color histogram (96 features, 32 bins x 3 channels)
- Preprocessing: Non-Local Means Denoising
- Training Data: Synthetic coal imagery

coal grade:
- grade_a : Highest quality coal, high carbon content, minimal impurities
- grade_b : Medium quality coal, moderate carbon content
- grade_c : Lower quality coal, visible impurities present
- reject  : Not suitable for industrial use, fails quality standards

Quality Hierarchy:
grade_a > grade_b > grade_c > reject
"""

# prompt
@app.prompt('classify with reasoning')
def classify_with_reasoning() -> str:
    """User-invoked prompt that instructs the LLM to classify 
    a coal image and explain the reasoning behind the prediction."""
    return """
You are a coal quality analysis assistant.

When classifying a coal image:
1. Call grade_coal() with the provided image
2. State the predicted grade clearly
3. Explain WHY this grade was assigned based on coal quality indicators
4. Reference the quality hierarchy (grade_a > grade_b > grade_c > reject)
5. Flag if the result should be verified by a domain expert

Format your response as:
- Predicted Grade: 
- Quality Level:
- Reasoning:
- Recommendation:
"""