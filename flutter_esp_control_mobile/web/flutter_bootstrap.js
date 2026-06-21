// web/flutter_bootstrap.js

{{flutter_js}}
{{flutter_build_config}}

_flutter.loader.load({
  config: {
    // 關鍵：不要走 CDN，強制走本站相對路徑
    // build/web 內會有 canvaskit/ 目錄（release build 時）
    canvasKitBaseUrl: 'canvaskit/',

    // 你也可以固定 chromium 變體（Edge/Chrome）省一點檔案
    // canvasKitVariant: 'chromium',

    // 建議先固定 canvaskit，比較不容易踩到 skwasm 的 thread/headers 相容問題
    renderer: 'canvaskit',
  },
});
