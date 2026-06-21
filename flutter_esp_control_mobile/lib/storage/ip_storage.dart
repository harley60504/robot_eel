import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class IpStorage {
  static const String _keyLastIp = "last_ip";
  static const String _keySsidIpMap = "ssid_ip_map";

  // ============================
  // last_ip
  // ============================
  static Future<void> saveLastIp(String ip) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyLastIp, ip);
  }

  static Future<String?> loadLastIp() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyLastIp);
  }

  static Future<void> clearLastIp() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keyLastIp);
  }

  // ============================
  // SSID -> IP map
  // ============================
  static Future<Map<String, String>> loadSsidIpMap() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_keySsidIpMap);

    if (raw == null || raw.isEmpty) return {};

    try {
      final data = jsonDecode(raw) as Map<String, dynamic>;
      return data.map((k, v) => MapEntry(k, v.toString()));
    } catch (_) {
      return {};
    }
  }

  static Future<void> saveIpForSsid(String ssid, String ip) async {
    final prefs = await SharedPreferences.getInstance();
    final map = await loadSsidIpMap();
    map[ssid] = ip;
    await prefs.setString(_keySsidIpMap, jsonEncode(map));
  }

  static Future<String?> loadIpForSsid(String ssid) async {
    final map = await loadSsidIpMap();
    return map[ssid];
  }

  static Future<void> deleteSsid(String ssid) async {
    final prefs = await SharedPreferences.getInstance();
    final map = await loadSsidIpMap();
    map.remove(ssid);
    await prefs.setString(_keySsidIpMap, jsonEncode(map));
  }

  static Future<void> clearSsidIpMap() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_keySsidIpMap);
  }

  static Future<void> clearAll() async {
    await clearLastIp();
    await clearSsidIpMap();
  }
}
