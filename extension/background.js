// Service worker: injects the reader into a tab on demand and relays its commands to the
// native host. One tab owns the helper at a time; a new reader supersedes the old one.
const HOST = "com.leovicenzi.read_aloud";
const IDLE_MS = 5 * 60 * 1000;

let native = null;
let owner = null; // the content-script port whose session the helper is serving
let lastId = 0; // the owner's latest command id, to stop its audio if the tab goes away
let idleTimer = null;

function explain(message) {
  if (/not found/i.test(message || "")) {
    return "Read-aloud host is not installed: run scripts/install.sh";
  }
  return `Read-aloud host stopped (${message || "exited"}); see ~/.cache/read-aloud-chromium/helper.log`;
}

function nativePort() {
  if (native) return native;
  native = chrome.runtime.connectNative(HOST);
  native.onMessage.addListener((event) => owner?.postMessage(event));
  // Only fires when the host ends on its own; release() disconnects deliberately.
  native.onDisconnect.addListener(() => {
    const message = chrome.runtime.lastError?.message;
    native = null;
    owner?.postMessage({ event: "released" });
    owner?.postMessage({ event: "error", id: 0, message: explain(message) });
  });
  return native;
}

// Closing the port makes Chromium stop the host, which stops the helper and frees its model.
function release() {
  clearTimeout(idleTimer);
  idleTimer = null;
  if (native) {
    native.disconnect();
    native = null;
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== "read-aloud") return;
  port.onMessage.addListener((message) => {
    if (message.type === "command") {
      if (owner !== port) {
        owner?.postMessage({ event: "superseded" });
        owner = port;
      }
      clearTimeout(idleTimer);
      lastId = message.command.id;
      nativePort().postMessage(message.command);
    } else if (message.type === "release" && owner === port) {
      port.postMessage({ event: "released" });
      release();
    }
  });
  port.onDisconnect.addListener(() => {
    if (owner !== port) return;
    // The tab closed or navigated: silence it, and keep the warm helper a while for the
    // next page.
    owner = null;
    if (native) {
      native.postMessage({ cmd: "stop", id: lastId });
      idleTimer = setTimeout(release, IDLE_MS);
    }
  });
});

async function activate(tab, mode) {
  if (!tab?.id) return;
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["text.js", "session.js", "content.js"],
    });
    await chrome.tabs.sendMessage(tab.id, { type: "activate", mode });
  } catch {
    // chrome:// pages, the Web Store, and the built-in PDF viewer refuse scripts.
    chrome.action.setBadgeText({ tabId: tab.id, text: "×" });
    setTimeout(() => chrome.action.setBadgeText({ tabId: tab.id, text: "" }), 3000);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "selection", title: "Read selection aloud", contexts: ["selection"] });
  chrome.contextMenus.create({ id: "page", title: "Read this page aloud", contexts: ["page"] });
});

chrome.contextMenus.onClicked.addListener((info, tab) => activate(tab, info.menuItemId));
chrome.action.onClicked.addListener((tab) => activate(tab, "auto"));
