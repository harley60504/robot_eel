import 'package:flutter/foundation.dart';
import 'package:network_info_plus/network_info_plus.dart';

class WifiInfo {
  static String? _bootSsid;
  static String? get bootSsid => _bootSsid;

  /// ✅ 開 App 時呼叫：存下 boot ssid
  static Future<void> initBootSsid() async {
    _bootSsid = await getCurrentSsid();
    print("[WifiInfo] bootSsid=$_bootSsid");
  }

  /// ✅ 真正去跟手機讀 SSID（含 retry）
  static Future<String?> getCurrentSsid() async {
    if (kIsWeb) return null;

    final info = NetworkInfo();

    for (int i = 0; i < 5; i++) {
      String? ssid = await info.getWifiName();

      if (ssid != null) {
        ssid = ssid.replaceAll('"', '').trim();
        if (ssid.isNotEmpty && !ssid.toLowerCase().contains("unknown")) {
          return ssid;
        }
      }

      await Future.delayed(const Duration(milliseconds: 250));
    }

    return null;
  }
}
