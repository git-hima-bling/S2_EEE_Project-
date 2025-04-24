document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('fault-detection-form');
    const initialMessage = document.getElementById('initial-message');
    const alertResults = document.getElementById('alert-results');
    const faultStatus = document.getElementById('fault-status');
    const faultType = document.getElementById('fault-type');
    const confidence = document.getElementById('confidence');
    const timestamp = document.getElementById('timestamp');
    const statusIndicator = document.getElementById('status-indicator');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        // Show loading state
        initialMessage.style.display = 'none';
        alertResults.style.display = 'block';
        faultStatus.textContent = 'Analyzing...';

        // Get form data
        const formData = new FormData(form);

        // Send request to server
        fetch('/detect_fault', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Update UI with results
            if (data.fault_detected) {
                faultStatus.textContent = 'FAULT DETECTED';
                statusIndicator.className = 'status-fault';
            } else {
                faultStatus.textContent = 'SYSTEM NORMAL';
                statusIndicator.className = 'status-ok';
            }

            faultType.textContent = data.fault_type;
            confidence.textContent = data.confidence;
            timestamp.textContent = data.timestamp;
        })
        .catch(error => {
            faultStatus.textContent = 'Error';
            faultType.textContent = error.message;
            statusIndicator.className = 'status-fault';
        });
    });

    // Tab switching functionality
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
        });
    });
});
