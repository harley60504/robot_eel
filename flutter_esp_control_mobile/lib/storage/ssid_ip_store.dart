import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class SsidIpStore {
  static const String _key = "ssid_ip_map";

  static Future<Map<String, String>> loadMap() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null || raw.isEmpty) return {};

    final decoded = jsonDecode(raw) as Map<String, dynamic>;
    return decoded.map((k, v) => MapEntry(k, v.toString()));
  }

  static Future<void> setIpForSsid(String ssid, String ip) async {
    final map = await loadMap();
    map[ssid] = ip;

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(map));
  }

  static Future<String?> getIpBySsid(String ssid) async {
    final map = await loadMap();
    return map[ssid];
  }
}
