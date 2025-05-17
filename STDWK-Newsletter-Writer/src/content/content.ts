// Content script for form detection and filling
console.log("ZenFill content script loaded");

// Listen for messages from the popup or background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "fillForm") {
    // Form filling logic will be implemented here
    console.log("Fill form request received");
  }
});
