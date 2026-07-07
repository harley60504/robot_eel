import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../config.dart';

class EspHttpApi {
  // ESP32 AP 預設 IP
  static String get baseUrl => ApiConfig.httpBaseUrl;

  /* ============================
   * 目前 Wi-Fi 狀態
   * GET /wifi_current
   * ============================ */
  static Future<Map<String, dynamic>> wifiCurrent() async {
    final res = await http.get(Uri.parse('$baseUrl/wifi_current'));

    if (res.statusCode != 200) {
      throw Exception('wifi_current failed');
    }

    return jsonDecode(res.body) as Map<String, dynamic>;
  }

  /* ============================
   * 已儲存 Wi-Fi
   * GET /wifi_saved
   * ============================ */
  static Future<List<String>> wifiSaved() async {
    final res = await http.get(Uri.parse('$baseUrl/wifi_saved'));

    if (res.statusCode != 200) {
      throw Exception('wifi_saved failed');
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final list = data['list'] as List;

    return list.map((e) => e['ssid'].toString()).toList();
  }

  /* ============================
   * 掃描附近 Wi-Fi
   * GET /wifi_scan
   * ============================ */
  static Future<List<Map<String, dynamic>>> wifiScan() async {
    final res = await http.get(Uri.parse('$baseUrl/wifi_scan'));

    if (res.statusCode != 200) {
      throw Exception('wifi_scan failed');
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final list = data['list'] as List;

    return List<Map<String, dynamic>>.from(list);
  }

  /* ============================
  * 連線 Wi-Fi 並儲存
  * GET /wifi_connect?ssid=xxx&pass=yyy
  * ============================ */
  static Future<bool> wifiConnect(String ssid, String pass) async {
    final uri = Uri.parse(
      '$baseUrl/wifi_connect',
    ).replace(queryParameters: {'ssid': ssid, 'pass': pass});

    final res = await http.get(uri);

    if (res.statusCode != 200) {
      throw Exception('wifi_connect failed');
    }

    return res.body.trim() == 'OK';
  }

  /* ============================
   * 刪除已儲存 Wi-Fi
   * GET /wifi_delete?ssid=xxx
   * ============================ */
  static Future<bool> wifiDelete(String ssid) async {
    final uri = Uri.parse(
      '$baseUrl/wifi_delete',
    ).replace(queryParameters: {'ssid': ssid});

    final res = await http.get(uri);

    if (res.statusCode != 200) {
      throw Exception('wifi_delete failed');
    }

    return res.body.trim() == 'OK';
  }

  static Future<Uint8List> servoLogCsv() async {
    final res = await http
        .get(Uri.parse('$baseUrl/servo_log.csv'))
        .timeout(const Duration(seconds: 45));

    if (res.statusCode != 200) {
      throw Exception('servo_log.csv failed');
    }

    return res.bodyBytes;
  }

  static Future<int> servoLogStatus() async {
    final res = await http
        .get(Uri.parse('$baseUrl/servo_log_status'))
        .timeout(const Duration(seconds: 5));

    if (res.statusCode != 200) {
      throw Exception('servo_log_status failed');
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final samples = data['samples'];
    return samples is num ? samples.toInt() : 0;
  }

  static Future<bool> clearServoLog() async {
    final res = await http
        .get(Uri.parse('$baseUrl/servo_log_clear'))
        .timeout(const Duration(milliseconds: 900));

    if (res.statusCode != 200) {
      throw Exception('servo_log_clear failed');
    }

    return res.body.trim() == 'OK';
  }
}
