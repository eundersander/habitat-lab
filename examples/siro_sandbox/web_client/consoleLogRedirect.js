const originalConsoleLog = console.log;
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;

function appendToLog(message, isError = false, isWarn = false) {
    const logOutput = document.getElementById("logOutput");
    const entry = document.createElement('div');

    if (isError) {
        entry.classList.add('error');
    } else if (isWarn) {
        entry.classList.add('warn');
    }

    entry.textContent = message;
    logOutput.appendChild(entry);

    // Scroll to the bottom of the div
    logOutput.scrollTop = logOutput.scrollHeight;
}

function customLog(...args) {
    // Log to the console
    originalConsoleLog(...args);
    
    // Log to the div
    const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' ');
    appendToLog(message);
}

function customWarn(...args) {
    // Log to the console
    originalConsoleWarn(...args);
    
    // Log to the div with error style
    const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' ');
    appendToLog(message, false, true);
}

function customError(...args) {
    // Log to the console
    originalConsoleError(...args);
    
    // Log to the div with error style
    const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' ');
    appendToLog(message, true);
}


// Override the original functions
console.log = customLog;
console.error = customError;
console.warn = customWarn;
