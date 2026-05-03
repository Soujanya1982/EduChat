'use strict';

// Tell Chrome to open the side panel whenever the toolbar icon is clicked.
// This is the correct MV3 API — more reliable than action.onClicked + open().
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch(console.error);
