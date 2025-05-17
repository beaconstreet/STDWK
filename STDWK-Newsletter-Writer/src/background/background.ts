// Background service worker for ZenFill
console.log("ZenFill background service worker loaded");

// Initialize extension
chrome.runtime.onInstalled.addListener(() => {
  console.log("ZenFill extension installed");
});

// Handle messages from content script or popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "fillForm") {
    // Forward the message to the content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, message);
      }
    });
  }
});
