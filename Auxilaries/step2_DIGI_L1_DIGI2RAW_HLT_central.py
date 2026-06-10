<!DOCTYPE html>
<html lang="en">
  <head>
    <base href="/">

    <script>window.WEB_APPS_MAP = {"web-app-activities":"./web-app-activities-h77Uwsbm.mjs","web-app-admin-settings":"./web-app-admin-settings-DJBURnBR.mjs","web-app-epub-reader":"./web-app-epub-reader-DWvxs8qd.mjs","web-app-external":"./web-app-external-xKDS2H7t.mjs","web-app-files":"./web-app-files-BM41RaBa.mjs","web-app-ocm":"./web-app-ocm-BRJxYZ6G.mjs","web-app-password-protected-folders":"./web-app-password-protected-folders-CMho1O9e.mjs","web-app-pdf-viewer":"./web-app-pdf-viewer-BmmeoZYG.mjs","web-app-preview":"./web-app-preview-BLSJ3o4i.mjs","web-app-search":"./web-app-search-CRANOKu-.mjs","web-app-text-editor":"./web-app-text-editor-CfZ-tlcY.mjs","web-app-webfinger":"./web-app-webfinger-B_5X3-GK.mjs","web-app-app-store":"./web-app-app-store-D6PJjDIK.mjs"}</script>

    <meta charset="utf-8" />
    <meta name="viewport" content="initial-scale=1.0, minimum-scale=1.0" />
    <meta name="theme-color" content="#375f7E" />
    <meta http-equiv="x-ua-compatible" content="IE=edge" />

    <title>CERNBox</title>
    <link rel="manifest" href="manifest.json" crossorigin="use-credentials" />
    
    <script src="js/require.js?1780905367662"></script>
    
    <style>
      html,
      body {
        height: 100%;
      }
      .splash-banner {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 0.5rem;
        height: 100%;
      }
      .splash-hide {
        display: none;
      }
      #loading {
        display: inline-block;
        height: 34px;
        width: 34px;
        border: 1px solid #4c5f79;
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
        -webkit-animation: spin 1s linear infinite;
      }
      #splash-incompatible button {
        margin: 30px 0;
      }

      @keyframes spin {
        to {
          -webkit-transform: rotate(360deg);
        }
      }
      @-webkit-keyframes spin {
        to {
          -webkit-transform: rotate(360deg);
        }
      }
    </style>
    <script type="module" crossorigin src="./js/index.html-CBW44v21.mjs"></script>
    <link rel="modulepreload" crossorigin href="./js/chunks/PortalTarget.vue_vue_type_script_lang-DSWY8v7Q.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useRouteMeta-Cw-i4i9W.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/ActionMenuItem-t6U72AZ_.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/SpaceInfo-CZJa_EEe.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/datetime-JPauBaAM.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/Pagination-DGsK-iwi.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useScrollTo-Bc-_D0t-.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/call-DmED1Wyl.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/api-CXt-nN61.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useLinkTypes-tz4CtqPo.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/FileSideBar-DXNVq-VD.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/omit-zLj_6BH7.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useGroupingSettings-q5LDwQ3e.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/AppLoadingSpinner-iTq2pK_T.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useAuthService-CM05q3Mo.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/isEmpty-vkZJc8E2.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useOpenEmptyEditor-DtKIiUvZ.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useAppDefaults-D-q2ZhU6.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/AppWrapperRoute-BjqmjsxQ.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/useAppProviderService-Dbx4JmAs.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/types-BoCZvwvE.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/CompareSaveDialog-BuWgBYI4.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/index-BRUAMNvw.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/fuse-Cqy8O5rp.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/ItemFilter-b5WR4LFX.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/NoContentMessage-DWE_6nRG.mjs">
    <link rel="modulepreload" crossorigin href="./js/chunks/SearchBarFilter-D2BbibJZ.mjs">
    <link rel="stylesheet" crossorigin href="./assets/style-PncNIgKq.css">
  </head>
  <body>
    <div id="splash-incompatible" class="splash-banner splash-hide">
      <div class="oc-card oc-border oc-rounded oc-width-large oc-text-center">
        <div class="oc-card-header">
          <div class="oc-flex oc-flex-middle oc-flex-center">
            <span class="oc-mr-s oc-icon oc-icon-m oc-icon-warning">
              <svg
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
                focusable="false"
              >
                <g xmlns="http://www.w3.org/2000/svg">
                  <path fill="none" d="M0 0h24v24H0z"></path>
                  <path
                    d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm0-2a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM11 7h2v2h-2V7zm0 4h2v6h-2v-6z"
                  ></path>
                </g>
              </svg>
            </span>
            <h2>Your browser is not supported</h2>
          </div>
        </div>
        <div class="oc-card-body oc-link-resolve-error-message">
          <p>Your browser version is considered old and might not work correctly.</p>
          <p>We recommend you update to a newer version.</p>
        </div>
      </div>
      <button
        class="oc-button oc-button-primary oc-button-primary-filled oc-rounded"
        onclick="forceOldBrowser()"
      >
        I want to continue anyway
      </button>
      
      <p>
        <a href="https://cernbox.docs.cern.ch/web/#desktop-requirements" target="_blank"
          >Click here to know more</a
        >
      </p>
      
    </div>
    <div id="splash-loading" class="splash-banner splash-hide">
      <div id="loading"></div>
    </div>
    <div id="owncloud"></div>
    <noscript>
      <div class="splash-banner"><h3>Please enable JavaScript</h3></div>
    </noscript>
    <script>
      function runtimeLoaded() {}

      var loader = document.getElementById('splash-loading')
      var browserError = document.getElementById('splash-incompatible')

      var loaderTimer = setTimeout(function () {
        loader.classList.remove('splash-hide')
      }, 500);

      function displayError() {
        loader.classList.remove('splash-hide')
        loader.innerHTML = "<h3>Oops. Something went wrong.</h3>"
      }

      function displayBrowserError() {
        clearTimeout(loaderTimer)
        removeLoadingSpinner()
        browserError.classList.remove('splash-hide')
      }

      function forceOldBrowser() {
        localStorage.setItem("forceAllowOldBrowser", JSON.stringify({expiry: new Date().getTime() + 30*24*60*60*1000}))
        browserError.classList.add('splash-hide')
        init()
      }

      function removeLoadingSpinner() {
        if (!loader.classList.contains('splash-hide')) {
          loader.classList.add('splash-hide')
        }
      }

      function init() {
        if (typeof requirejs === 'undefined') {
          displayError()
        } else {
          window.runtimeLoaded = function(runtime) {
            clearTimeout(loaderTimer)
            runtime.bootstrapApp('config.json', removeLoadingSpinner).catch((error) => {
              removeLoadingSpinner()
              runtime.bootstrapErrorApp(error)
            })
          }
        }
      }

      const supportedBrowsers = /Edge?\/(13[1-9]|1[4-9]\d|[2-9]\d{2}|\d{4,})\.\d+(\.\d+|)|Firefox\/(1{2}[5-9]|1[2-9]\d|[2-9]\d{2}|\d{4,})\.\d+(\.\d+|)|Chrom(ium|e)\/(109|1[1-9]\d|[2-9]\d{2}|\d{4,})\.\d+(\.\d+|)|(Maci|X1{2}).+ Version\/(16\.([6-9]|\d{2,})|(1[7-9]|[2-9]\d|\d{3,})\.\d+)([,.]\d+|)( \(\w+\)|)( Mobile\/\w+|) Safari\/|Chrome.+OPR\/(1{2}[4-9]|1[2-9]\d|[2-9]\d{2}|\d{4,})\.\d+\.\d+|(CPU[ +]OS|iPhone[ +]OS|CPU[ +]iPhone|CPU IPhone OS|CPU iPad OS)[ +]+(15[._]([6-9]|\d{2,})|(1[6-9]|[2-9]\d|\d{3,})[._]\d+)([._]\d+|)|Android:?[ /-](13[2-9]|1[4-9]\d|[2-9]\d{2}|\d{4,})(\.\d+|)(\.\d+|)|Android.+Firefox\/(13[2-9]|1[4-9]\d|[2-9]\d{2}|\d{4,})\.\d+(\.\d+|)|Android.+Chrom(ium|e)\/(13[2-9]|1[4-9]\d|[2-9]\d{2}|\d{4,})\.\d+(\.\d+|)|Android.+(UC? ?Browser|UCWEB|U3)[ /]?(15\.([5-9]|\d{2,})|(1[6-9]|[2-9]\d|\d{3,})\.\d+)\.\d+|SamsungBrowser\/(2[7-9]|[3-9]\d|\d{3,})\.\d+/
      const forceAllowOldBrowser = localStorage.getItem("forceAllowOldBrowser") || false
      const validForceAllowOldBrowser = forceAllowOldBrowser && JSON.parse(localStorage.getItem("forceAllowOldBrowser")).expiry > new Date().getTime()

      if (forceAllowOldBrowser && !validForceAllowOldBrowser)
        localStorage.removeItem("forceAllowOldBrowser")

      if (!validForceAllowOldBrowser && !supportedBrowsers.test(navigator.userAgent)) {
        displayBrowserError()
      } else {
        init()
      }

      var scriptTags = document.getElementsByTagName('script')
      for (let i = 0; i < scriptTags.length; i++) {
        if (scriptTags[i].src) {
          scriptTags[i].onerror = displayError
        }
      }
    </script>
  </body>
</html>
