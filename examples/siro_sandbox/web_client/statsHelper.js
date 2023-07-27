// statsHelper.js

// Track received message count and rate
let receivedCount = 0; // we rely on other scripts updating this
let lastReceivedCount = 0;
let startTime = Date.now();

// Start updating the message rate and received count once per second
let updateIntervalId = setInterval(updateStats, 1000);

// Function to update message rate and received count
function updateStats() {
    const currentTime = Date.now();
    const elapsedTime = (currentTime - startTime) / 1000; // Convert to seconds
    const messageRate = (receivedCount - lastReceivedCount) / elapsedTime;

    // Update HTML elements with message rate and received count
    document.getElementById("messageRate").textContent = messageRate.toFixed(2);
    document.getElementById("receivedCount").textContent = receivedCount;

    lastReceivedCount = receivedCount;
    startTime = currentTime;
}
