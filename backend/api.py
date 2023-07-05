from flask import Flask, request, jsonify
import main 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Check if a file is present in the request
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    image_file = request.files['image']
    print("image_file", image_file)
    
    # Check if the file has an allowed extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return 'Invalid file extension', 400

    # Save the uploaded file to a temporary location
    temp_image_path = "D:/Image-description/backend/temp-image/temp_image.jpg"  # Provide the desired path to save the temporary image
    image_file.save(temp_image_path)
    print("here")
    
    image_path_array = [temp_image_path]
    # Process the uploaded image using the ImageCaptioning class
    captions = main.get_captions(image_path_array)

    # Remove the temporary image file
    # Uncomment the following line if you want to delete the temporary image
    # os.remove(temp_image_path)

    # Return the generated captions as a JSON response
    response = {'captions': captions}
    return response

if __name__ == '__main__':
    app.run(debug=True)




