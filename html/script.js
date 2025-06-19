document.addEventListener('DOMContentLoaded', () => {
    const csvFileInput = document.getElementById('csvFile');
    const startTrainingBtn = document.getElementById('startTrainingBtn');
    const statusMessageDiv = document.getElementById('statusMessage');
    const apiKeySection = document.getElementById('apiKeySection');
    const apiKeyDisplay = document.getElementById('apiKeyDisplay');
    const modelPathDisplay = document.getElementById('modelPathDisplay');
    const mappingsPathDisplay = document.getElementById('mappingsPathDisplay');

    startTrainingBtn.addEventListener('click', async () => {
        const file = csvFileInput.files[0];
        if (!file) {
            displayMessage('Please select a CSV file.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('training_data', file);

        displayMessage('Training started... This may take a few minutes.', 'info', true);
        apiKeySection.style.display = 'none'; // Hide previous key if any

        try {
            // Assuming the FastAPI server is running on http://localhost:8000
            // Adjust if your server URL is different.
            const response = await fetch('/v1/train', {
                method: 'POST',
                body: formData,
                // Headers are not strictly necessary for FormData with fetch,
                // but shown for completeness if issues arise.
                // headers: { 'accept': 'application/json' }
            });

            startTrainingBtn.disabled = false; // Re-enable button

            if (response.ok) {
                const result = await response.json();
                displayMessage('Training successful!', 'success');
                apiKeyDisplay.textContent = result.api_key;
                modelPathDisplay.textContent = result.model_path;
                mappingsPathDisplay.textContent = result.mappings_path;
                apiKeySection.style.display = 'block';
            } else {
                const errorResult = await response.json();
                let errorMessage = `Training failed. Status: ${response.status}`;
                if (errorResult && errorResult.detail) {
                    if (Array.isArray(errorResult.detail)) { // Handle FastAPI validation errors
                        errorMessage += "\nDetails:\n" + errorResult.detail.map(err => `- ${err.loc.join(" -> ")}: ${err.msg}`).join("\n");
                    } else {
                         errorMessage += `\nDetails: ${errorResult.detail}`;
                    }
                } else {
                    errorMessage += `\nServer response: ${await response.text()}`;
                }
                displayMessage(errorMessage, 'error');
                apiKeySection.style.display = 'none';
            }
        } catch (error) {
            console.error('Error during training:', error);
            displayMessage(`An error occurred: ${error.message}. Check the console for details.`, 'error');
            startTrainingBtn.disabled = false;
            apiKeySection.style.display = 'none';
        }
    });

    function displayMessage(message, type, isLoading = false) {
        statusMessageDiv.textContent = message;
        statusMessageDiv.className = type; // 'success', 'error', or 'info'
        startTrainingBtn.disabled = isLoading;
    }
});
