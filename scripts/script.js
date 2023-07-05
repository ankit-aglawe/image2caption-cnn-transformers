
document.getElementById('uploadForm').addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent default form submission
  event.stopPropagation(); // Stop event propagation

  var formData = new FormData();
  var fileInput = document.getElementById('imageInput');

  formData.append('image', fileInput.files[0]);

  document.getElementById('loading').style.display = 'block'; // Show loading animation

  // Show image preview immediately after selecting a file
  document.getElementById('imagePreview').style.display = 'block';
  document.getElementById('previewImage').src = URL.createObjectURL(fileInput.files[0]);

  fetch('http://127.0.0.1:5000/upload-image', {
    method: 'POST',
    body: formData
  })
  .then(function(response) {
    if (response.ok) {
      return response.json();
    } else {
      throw new Error('Image upload failed.');
    }
  })
  .then(function(data) {
    var caption = data.captions[0]; // Get the first caption from the response
    document.getElementById('result').innerHTML = 'Caption Generated: "' + caption + '"';
    document.getElementById('uploadForm').style.display = 'none'; // Hide upload form
    document.getElementById('loading').style.display = 'none'; // Hide loading animation
    document.getElementById('anotherImageBtn').style.display = 'block'; // Show "Another Image" button
    // Process the response data if needed
  })
  .catch(function(error) {
    console.error('Error:', error);
    document.getElementById('result').innerHTML = 'Image upload failed.';
    document.getElementById('loading').style.display = 'none'; // Hide loading animation
  });
});

document.getElementById('anotherImageBtn').addEventListener('click', function() {
  window.location.href = 'index.html'; // Replace 'index.html' with your initial page URL
});