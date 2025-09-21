// Store the ID of the popup window when it's opened
let popupWindowId = null;

// Listen for messages from Detector.jsx for stress detection, badge, and notifications
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'stressDetected') {
        // Show a notification when stress is detected
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'images/icon128.png', // Replace with your icon path
            title: 'Stress Detector Alert',
            message: 'Stress detected! Please take a break and relax.',
            priority: 2
        });

        // Set a badge on the extension icon indicating stress detection
        chrome.action.setBadgeText({ text: '!' });
        chrome.action.setBadgeBackgroundColor({ color: '#FF0000' });

        sendResponse({ status: 'Notification sent and badge updated' });
    } else if (request.action === 'clearBadge') {
        // Clear the badge when the popup is opened
        console.log("Clearing badge...");
        chrome.action.setBadgeText({ text: '' });
        sendResponse({ status: 'Badge cleared' });
    } else if (request.action === 'startRecording') {
        console.log("Recording started");
    } else if (request.action === 'stopRecording') {
        console.log("Recording stopped");
    }
});

// Handle notification click to bring the existing popup window to the front or open a new one if it doesn't exist
chrome.notifications.onClicked.addListener(() => {
    clearBadgeAndOpenPopup();
});

// Handle extension icon click to open the popup or bring the existing one to the front
chrome.action.onClicked.addListener(() => {
    clearBadgeAndOpenPopup();
});

// Clear badge and open the popup window or bring the existing one to the front
function clearBadgeAndOpenPopup() {
    // Clear the badge
    chrome.action.setBadgeText({ text: '' });
    console.log("Badge cleared after interaction");

    // Bring the popup window to the front or open a new one
    if (popupWindowId !== null) {
        chrome.windows.update(popupWindowId, { focused: true }, (window) => {
            if (chrome.runtime.lastError || !window) {
                popupWindowId = null;
                openPopupWindow();
            }
        });
    } else {
        openPopupWindow();
    }
}

// Function to actually create the popup window
function openPopupWindow() {
    chrome.windows.create({
        url: chrome.runtime.getURL("popup.html"),
        type: "popup",
        width: 800,
        height: 600,
        focused: true
    }, (window) => {
        popupWindowId = window.id;
    });
}

// Handle when a window is closed and clear the popupWindowId if it's the popup window
chrome.windows.onRemoved.addListener((windowId) => {
    if (windowId === popupWindowId) {
        popupWindowId = null; // Reset popupWindowId when the popup is closed
    }
});
