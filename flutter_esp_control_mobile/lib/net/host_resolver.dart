import 'dart:async';
import 'package:http/http.dart' as http;

import '../storage/ip_storage.dart';
import 'wifi_info.dart';

class HostResult {
  final String host;
  final String reason; // ap_ssid / ssid_map / last_ip / ap_fixed
  const HostResult(this.host, this.reason);
}

class HostResolver {
  static const String apHost = "192.168.4.1";

  /// ESP32 AP SSID 名稱（可多台）
  static const List<String> apSsidList = [
    "robot",
  ];

  static bool isEspApSsid(String ssid) {
    final s = ssid.trim().toLowerCase();
    return apSsidList.any((ap) => s == ap.toLowerCase());
  }

  /// HTTP ping
  static Future<bool> pingHost(String host) async {
    for (int i = 0; i < 2; i++) {
      try {
        final url = Uri.parse("http://$host/wifi_current");
        final res =
            await http.get(url).timeout(const Duration(milliseconds: 1500));
        if (res.statusCode == 200) return true;
      } catch (_) {}

      await Future.delayed(const Duration(milliseconds: 250));
    }
    return false;
  }

  /// ===== 主流程 =====
  static Future<HostResult> autoSelectHostEx() async {
    final ssid = WifiInfo.bootSsid ?? await WifiInfo.getCurrentSsid();
    final lastIp = await IpStorage.loadLastIp();

    print("[HostResolver] phone ssid = $ssid");
    print("[HostResolver] last_ip = $lastIp");

    // ------------------------------------------------------------
    // 0) ESP32 AP 模式
    // ------------------------------------------------------------
    if (ssid != null && isEspApSsid(ssid)) {
      print("[HostResolver] detected ESP AP ssid=$ssid -> $apHost");
      return HostResult(apHost, "ap_ssid($ssid)");
    }

    // ------------------------------------------------------------
    // 1) ssid -> ip map
    // ------------------------------------------------------------
    if (ssid != null) {
      final mappedIp = await IpStorage.loadIpForSsid(ssid);
      print("[HostResolver] ssid->ip = $mappedIp");

      if (mappedIp != null) {
        final ok = await pingHost(mappedIp);
        print("[HostResolver] ping mapped ip $mappedIp = $ok");

        if (ok) {
          return HostResult(mappedIp, "ssid_map($ssid)");
        }
      }
    }

    // ------------------------------------------------------------
    // 2) last_ip（⭐ 成功後「修復 ssid_map」）
    // ------------------------------------------------------------
    if (lastIp != null) {
      final ok = await pingHost(lastIp);
      print("[HostResolver] ping last_ip $lastIp = $ok");

      if (ok) {
        if (ssid != null && !isEspApSsid(ssid)) {
          await IpStorage.saveIpForSsid(ssid, lastIp);
          print("[HostResolver] repair ssid_map: $ssid -> $lastIp");
        }
        return HostResult(lastIp, "last_ip");
      }
    }

    // ------------------------------------------------------------
    // 3) fallback AP
    // ------------------------------------------------------------
    print("[HostResolver] use ap_fixed = $apHost");
    return const HostResult(apHost, "ap_fixed");
  }

  static Future<String> autoSelectHost() async {
    final r = await autoSelectHostEx();
    return r.host;
  }

  /// 給 WiFi 卡片用：STA 連線後更新 cache
  static Future<void> updateCachesByStaIp(String staIp) async {
    final lastIp = await IpStorage.loadLastIp();
    if (lastIp != staIp) {
      print("[HostResolver] IP changed: $lastIp -> $staIp");
    }

    await IpStorage.saveLastIp(staIp);

    final ssid = WifiInfo.bootSsid ?? await WifiInfo.getCurrentSsid();
    if (ssid != null && !isEspApSsid(ssid)) {
      await IpStorage.saveIpForSsid(ssid, staIp);
      print("[HostResolver] save map: $ssid -> $staIp");
    }
  }
}
