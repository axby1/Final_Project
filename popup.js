// this is the front end for the plugin
// note the backendpoint URL = http://localhost:3000/classify 
// always go to this page before using the plugin


// document.addEventListener('DOMContentLoaded', function () {
//     document.getElementById('classify-button').addEventListener('click', classifyUrl);
//   });
  
//   function classifyUrl() {
//     const urlInput = document.getElementById('url');
//     const resultContainer = document.getElementById('result-container');
  
//     const url = urlInput.value.trim();
  
//     if (url) {
     
//       sendUrlToBackend(url)
//         .then((response) => {
//           const category = response.category || 'Unknown';
//           resultContainer.innerHTML = `<p>Category: ${category}</p>`;
//         })
//         .catch((error) => {
//           console.error('Error processing the URL:', error);
//           resultContainer.innerHTML = '<p>Error processing the URL. Please try again.</p>';
//         });
//     } else {
//       resultContainer.innerHTML = '<p>Please enter a URL.</p>';
//     }
//   }
  
//   function sendUrlToBackend(url) {
//     // endpoint of backend
//     const backendEndpoint = 'http://localhost:3000/classify';
    
    
  
//     return fetch(backendEndpoint, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ url: url }),
//     })
//       .then((response) => {
//         if (!response.ok) {
//           throw new Error(`HTTP error! Status: ${response.status}`);
//         }
//         return response.json();
//       })
//       .catch((error) => {
//         throw new Error(`Error sending URL to backend: ${error.message}`);
//       });
//   }
  


// document.addEventListener('DOMContentLoaded', function () {
//   document.getElementById('classify-button').addEventListener('click', classifyUrl);
// });

// function classifyUrl() {
//   const urlInput = document.getElementById('url');
//   const resultContainer = document.getElementById('result-container');

//   const url = urlInput.value.trim();

//   if (url) {
//       sendUrlToBackend(url)
//           .then((response) => {
//               const category = response.category || 'Unknown';
//               let resultHtml = `<p>Category: ${category}</p>`;
              
//               // Check if the response contains AI response
              
//               const aiResponse = response.ai_response;
//               resultHtml += `<p>AI Response: ${aiResponse}</p>`;
              

//               resultContainer.innerHTML = resultHtml;
//           })
//           .catch((error) => {
//               console.error('Error processing the URL:', error);
//               resultContainer.innerHTML = '<p>Error processing the URL. Please try again.</p>';
//           });
//   } else {
//       resultContainer.innerHTML = '<p>Please enter a URL.</p>';
//   }
// }

// function sendUrlToBackend(url) {
//   // Endpoint of backend
//   const backendEndpoint = 'http://localhost:3000/classify';

//   return fetch(backendEndpoint, {
//       method: 'POST',
//       headers: {
//           'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ url: url }),
//   })
//   .then((response) => {
//       if (!response.ok) {
//           throw new Error(`HTTP error! Status: ${response.status}`);
//       }
//       return response.json();
//   })
//   .catch((error) => {
//       throw new Error(`Error sending URL to backend: ${error.message}`);
//   });
// }


// //correct version 
// document.addEventListener('DOMContentLoaded', function () {
//   document.getElementById('classify-button').addEventListener('click', classifyUrl);
// });

// function classifyUrl() {
//   const urlInput = document.getElementById('url');
//   const resultContainer = document.getElementById('result-container');

//   const url = urlInput.value.trim();

//   if (url) {
//       resultContainer.innerHTML = ''; // Clear previous results
      
//       sendUrlToBackend(url);
//   } else {
//       resultContainer.innerHTML = '<p>Please enter a URL.</p>';
//   }
// }

// function sendUrlToBackend(url) {
//   const backendEndpoint = 'http://localhost:3000/classify';

//   return fetch(backendEndpoint, {
//       method: 'POST',
//       headers: {
//           'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ url: url }),
//   })
//   .then((response) => {
//       if (!response.ok) {
//           throw new Error(`HTTP error! Status: ${response.status}`);
//       }
//       return response.json();
//   })
//   .then((data) => {
//       const category = data.category;
//       const resultContainer = document.getElementById('result-container');
//       resultContainer.innerHTML += `<p>Category: ${category}</p>`;
      
//       // Proceed to interact with AI model if the category is obtained
//       if (category) {
//           startAiInteraction();
//       }
//   })
//   .catch((error) => {
//       console.error('Error processing the URL:', error);
//       const resultContainer = document.getElementById('result-container');
//       resultContainer.innerHTML = '<p>Error processing the URL. Please try again.</p>';
//   });
// }

// function startAiInteraction() {
//   const resultContainer = document.getElementById('result-container');
//   const exitCommand = 'exit';
//   let query = prompt("You:");
  
//   while (query && query.trim().toLowerCase() !== exitCommand) {
//       resultContainer.innerHTML += `<p>You: ${query}</p>`;
      
//       sendQueryToBackend(query);
//       query = prompt("You:");
//   }
// }

// function sendQueryToBackend(query) {
//   const backendEndpoint = 'http://localhost:3000/query';

//   return fetch(backendEndpoint, {
//       method: 'POST',
//       headers: {
//           'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ query: query }),
//   })
//   .then((response) => {
//       if (!response.ok) {
//           throw new Error(`HTTP error! Status: ${response.status}`);
//       }
//       return response.json();
//   })
//   .then((data) => {
//       const aiResponse = data.ai_response;
//       const resultContainer = document.getElementById('result-container');
//       resultContainer.innerHTML += `<p>AI: ${aiResponse}</p>`;
//   })
//   .catch((error) => {
//       console.error('Error sending query to backend:', error);
//       const resultContainer = document.getElementById('result-container');
//       resultContainer.innerHTML += '<p>Error processing the query. Please try again.</p>';
//   });
// }




//latest version
document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('classify-button').addEventListener('click', classifyUrl);
    document.getElementById('ai-button').addEventListener('click', startAiInteraction);
});

function classifyUrl() {
    const urlInput = document.getElementById('url');
    const url = urlInput.value.trim();

    if (url) {
        sendUrlToBackend(url);
    } else {
        alert('Please enter a URL.');
    }
}

function sendUrlToBackend(url) {
    const backendEndpoint = 'http://localhost:3000/classify';

    fetch(backendEndpoint, {
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
    .then((data) => {
        const category = data.category;
        const resultContainer = document.getElementById('result-container');
        resultContainer.innerHTML = `<p>Category: ${category}</p>`;
    })
    .catch((error) => {
        console.error('Error processing the URL:', error);
        alert('Error processing the URL. Please try again.');
    });
}

function startAiInteraction() {
    const query = prompt("Enter your query:");
    if (query !== null && query.trim() !== '') {
        sendQueryToBackend(query);
    }
}

function sendQueryToBackend(query) {
    const backendEndpoint = 'http://localhost:3000/query';

    fetch(backendEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
    })
    .then((response) => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then((data) => {
        const aiResponse = data.ai_response;
        const resultContainer = document.getElementById('result-container');
        resultContainer.innerHTML += `<p>User: ${query}</p><p>AI Response: ${aiResponse}</p>`;
    })
    .catch((error) => {
        console.error('Error sending query to backend:', error);
        alert('Error processing the query. Please try again.');
    });
}
