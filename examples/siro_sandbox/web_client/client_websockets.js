// client.js

// this is wrong! don't vary the port; vary ws: vs wss:
const use_ssl = location.protocol === "https:";
const port = use_ssl ? "8433" : "8080";
const serverAddress = "ws://192.168.4.26:8888"; //  + port;

// Create a WebSocket connection
const socket = new WebSocket(serverAddress);

// Event handler for when the connection is established
socket.onopen = () => {
    console.log("Connected to server");

    // Start sending "marco" messages as fast as possible
    sendMarco();

    // Start updating the message rate and received count once per second
    updateIntervalId = setInterval(updateStats, 1000);
};

// Event handler for receiving messages
socket.onmessage = (event) => {
    const message = event.data;
    console.log("Received:", message);

    if (message === "polo") {
        // Wait for "polo" before sending the next "marco" message
        sendMarco();
    }

    // Increment received count
    receivedCount++;
};

// Event handler for when the connection is closed
socket.onclose = () => {
    console.log("Connection closed");

    // Stop updating the message rate and received count
    clearInterval(updateIntervalId);
};

// Event handler for errors
socket.onerror = (error) => {
    console.error("WebSocket error:", error);
};

// Function to send "marco" message
function sendMarco() {
    const message = "marco";
    socket.send(message);
    console.log("Sent:", message);
}
