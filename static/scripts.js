document.getElementById("submitBtn").addEventListener("click", function () {
    // Gather input values
    let feature1 = document.getElementById("feature1").value;
    let feature2 = document.getElementById("feature2").value;

    // Collect all features into an array
    let features = [parseFloat(feature1), parseFloat(feature2)];

    // Send data to the backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 'features': features })
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction) {
            document.getElementById("result").innerText = `Prediction: ${data.prediction}`;
        } else {
            document.getElementById("result").innerText = `Error: ${data.error}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
