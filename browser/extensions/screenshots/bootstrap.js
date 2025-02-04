/* globals ADDON_DISABLE Services CustomizableUI LegacyExtensionsUtils AppConstants PageActions */
const ADDON_ID = "screenshots@mozilla.org";
const TELEMETRY_ENABLED_PREF = "datareporting.healthreport.uploadEnabled";
const PREF_BRANCH = "extensions.screenshots.";
const USER_DISABLE_PREF = "extensions.screenshots.disabled";
const UPLOAD_DISABLED_PREF = "extensions.screenshots.upload-disabled";
const HISTORY_ENABLED_PREF = "places.history.enabled";

ChromeUtils.import("resource://gre/modules/XPCOMUtils.jsm");
ChromeUtils.defineModuleGetter(this, "AddonManager",
                               "resource://gre/modules/AddonManager.jsm");
ChromeUtils.defineModuleGetter(this, "AddonManagerPrivate",
                               "resource://gre/modules/AddonManager.jsm");
ChromeUtils.defineModuleGetter(this, "AppConstants",
                               "resource://gre/modules/AppConstants.jsm");
ChromeUtils.defineModuleGetter(this, "CustomizableUI",
                               "resource:///modules/CustomizableUI.jsm");
ChromeUtils.defineModuleGetter(this, "LegacyExtensionsUtils",
                               "resource://gre/modules/LegacyExtensionsUtils.jsm");
ChromeUtils.defineModuleGetter(this, "Services",
                               "resource://gre/modules/Services.jsm");

let addonResourceURI;
let appStartupDone;
let appStartupPromise = new Promise((resolve, reject) => {
  appStartupDone = resolve;
});

const prefs = Services.prefs;

const appStartupObserver = {
  register() {
    Services.obs.addObserver(this, "sessionstore-windows-restored", false); // eslint-disable-line mozilla/no-useless-parameters
  },

  unregister() {
    Services.obs.removeObserver(this, "sessionstore-windows-restored", false); // eslint-disable-line mozilla/no-useless-parameters
  },

  observe() {
    appStartupDone();
    this.unregister();
  }
};

const LibraryButton = {
  ITEM_ID: "appMenu-library-screenshots",

  init(webExtension) {
    this._initialized = true;
    const permissionPages = [...webExtension.extension.permissions].filter(p => (/^https?:\/\//i).test(p));
    if (permissionPages.length > 1) {
      Cu.reportError(new Error("Should not have more than 1 permission page, but got: " + JSON.stringify(permissionPages)));
    }
    this.PAGE_TO_OPEN = permissionPages.length === 1 ? permissionPages[0].replace(/\*$/, "") : "https://screenshots.firefox.com/";
    this.PAGE_TO_OPEN += "shots";
    this.ICON_URL = webExtension.extension.getURL("icons/icon-v2.svg");
    this.LABEL = webExtension.extension.localizeMessage("libraryLabel");
    CustomizableUI.addListener(this);
    for (const win of CustomizableUI.windows) {
      this.onWindowOpened(win);
    }
  },

  uninit() {
    if (!this._initialized) {
      return;
    }
    for (const win of CustomizableUI.windows) {
      const item = win.document.getElementById(this.ITEM_ID);
      if (item) {
        item.remove();
      }
    }
    CustomizableUI.removeListener(this);
    this._initialized = false;
  },

  onWindowOpened(win) {
    const libraryViewInsertionPoint = win.document.getElementById("appMenu-library-remotetabs-button");
    // If the library view doesn't exist (on non-photon builds, for instance),
    // this will be null, and we bail out early.
    if (!libraryViewInsertionPoint) {
      return;
    }
    const parent = libraryViewInsertionPoint.parentNode;
    const {nextSibling} = libraryViewInsertionPoint;
    const item = win.document.createXULElement("toolbarbutton");
    item.className = "subviewbutton subviewbutton-iconic";
    item.addEventListener("command", () => win.openWebLinkIn(this.PAGE_TO_OPEN, "tab"));
    item.id = this.ITEM_ID;
    const iconURL = this.ICON_URL;
    item.setAttribute("image", iconURL);
    item.setAttribute("label", this.LABEL);

    parent.insertBefore(item, nextSibling);
  },
};

let addonData, startupReason;

function startup(data, reason) { // eslint-disable-line no-unused-vars
  addonResourceURI = data.resourceURI;

  if (Services.prefs.getBoolPref(USER_DISABLE_PREF, false)) {
    AddonManager.getActiveAddons().then(result => {
      let addon = result.addons.find(a => a.id == ADDON_ID);
      if (addon) {
        addon.disable({allowSystemAddons: true});
      }
    });
    return;
  }

  addonData = data;
  startupReason = reason;
  if (reason === AddonManagerPrivate.BOOTSTRAP_REASONS.APP_STARTUP) {
    appStartupObserver.register();
  } else {
    appStartupDone();
  }
  // eslint-disable-next-line promise/catch-or-return
  appStartupPromise = appStartupPromise.then(handleStartup);
}

function shutdown(data, reason) { // eslint-disable-line no-unused-vars
  const webExtension = LegacyExtensionsUtils.getEmbeddedExtensionFor({
    id: ADDON_ID,
    resourceURI: addonResourceURI
  });
  // Immediately exit if Firefox is exiting, #3323
  if (reason === AddonManagerPrivate.BOOTSTRAP_REASONS.APP_SHUTDOWN) {
    stop(webExtension, reason);
    return;
  }
  // Because the prefObserver is unregistered above, this _should_ terminate the promise chain.
  appStartupPromise = appStartupPromise.then(() => { stop(webExtension, reason); });
}

function install(data, reason) {} // eslint-disable-line no-unused-vars

function uninstall(data, reason) {} // eslint-disable-line no-unused-vars

function getBoolPref(pref) {
  return prefs.getPrefType(pref) && prefs.getBoolPref(pref);
}

function handleStartup() {
  const webExtension = LegacyExtensionsUtils.getEmbeddedExtensionFor({
    id: ADDON_ID,
    resourceURI: addonResourceURI
  });

  if (!webExtension.started) {
    start(webExtension);
  }
}

function start(webExtension) {
  let reasonStr = stringReasonFromNumericReason(startupReason);
  return webExtension.startup(reasonStr, addonData).then((api) => {
    api.browser.runtime.onMessage.addListener(handleMessage);
    LibraryButton.init(webExtension);
  }).catch((err) => {
    // The startup() promise will be rejected if the webExtension was
    // already started (a harmless error), or if initializing the
    // WebExtension failed and threw (an important error).
    console.error(err);
    if (err.message !== "This embedded extension has already been started") {
      // TODO: Should we send these errors to Sentry? #2420
    }
  });
}

function stop(webExtension, reason) {
  if (reason !== AddonManagerPrivate.BOOTSTRAP_REASONS.APP_SHUTDOWN) {
    LibraryButton.uninit();
  }
  let reasonStr = stringReasonFromNumericReason(reason);
  return Promise.resolve(webExtension.shutdown(reasonStr));
}

function stringReasonFromNumericReason(numericReason) {
  let { BOOTSTRAP_REASONS } = AddonManagerPrivate;
  return Object.keys(BOOTSTRAP_REASONS).find(
    key => BOOTSTRAP_REASONS[key] == numericReason
  );
}

function handleMessage(msg, sender, sendReply) {
  if (!msg) {
    return;
  }

  if (msg.funcName === "isTelemetryEnabled") {
    const telemetryEnabled = getBoolPref(TELEMETRY_ENABLED_PREF);
    sendReply({type: "success", value: telemetryEnabled});
  } else if (msg.funcName === "isUploadDisabled") {
    const isESR = AppConstants.MOZ_UPDATE_CHANNEL === "esr";
    const uploadDisabled = getBoolPref(UPLOAD_DISABLED_PREF);
    sendReply({type: "success", value: uploadDisabled || isESR});
  } else if (msg.funcName === "isHistoryEnabled") {
    const historyEnabled = getBoolPref(HISTORY_ENABLED_PREF);
    sendReply({type: "success", value: historyEnabled});
  } else if (msg.funcName === "incrementCount") {
    const allowedScalars = ["download", "upload", "copy"];
    const scalar = msg.args && msg.args[0] && msg.args[0].scalar;
    if (!allowedScalars.includes(scalar)) {
      sendReply({type: "error", name: `incrementCount passed an unrecognized scalar ${scalar}`});
    } else {
      Services.telemetry.scalarAdd(`screenshots.${scalar}`, 1);
      sendReply({type: "success", value: true});
    }
  }
}
