// this is the front end for the plugin
// note the backendpoint URL = http://localhost:3000/classify 
// always go to this page before using the plugin

document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('classify-button').addEventListener('click', classifyUrl);
  });
  
  function classifyUrl() {
    const urlInput = document.getElementById('url');
    const resultContainer = document.getElementById('response');
  
    const url = urlInput.value.trim();
  
    if (url) {
      // Send the URL to the backend for processing
      sendUrlToBackend(url)
        .then((response) => {
          const category = response.category || 'Unknown';
          resultContainer.innerHTML = `<p>Category: ${category}</p>`;
        })
        .catch((error) => {
          console.error('Error processing the URL:', error);
          resultContainer.innerHTML = '<p>Error processing the URL. Please try again.</p>';
        });
    } else {
      resultContainer.innerHTML = '<p>Please enter a URL.</p>';
    }
  }
  
  function sendUrlToBackend(url) {
    // Replace the URL with the actual endpoint of your backend
    const backendEndpoint = 'http://localhost:3000/classify';
    
    
  
    return fetch(backendEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .catch((error) => {
        throw new Error(`Error sending URL to backend: ${error.message}`);
      });
  }
  