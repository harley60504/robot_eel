import 'package:shared_preferences/shared_preferences.dart';

class ApiConfig {
  static String host = "192.168.4.1";
  static String hostReason = "unknown";
  static String pythonHost = "127.0.0.1";
  static int pythonPort = 8765;
  static String recorderUrl =
      "rtsp://admin:184342@192.168.0.102:554/live/profile.0/video";

  static String get httpBaseUrl => "http://$host";
  static String get wsControlUrl => "ws://$host:82";
  static String get wsStreamUrl => "ws://$host:81";
  static String get pythonBaseUrl => "http://$pythonHost:$pythonPort";

  static Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    pythonHost = prefs.getString("python_host") ?? pythonHost;
    pythonPort = prefs.getInt("python_port") ?? pythonPort;
    recorderUrl = prefs.getString("recorder_url") ?? recorderUrl;
  }

  static Future<void> setHost(String newHost,
      {String reason = "manual"}) async {
    host = newHost;
    hostReason = reason;
  }

  static Future<void> setPythonHost(String newHost) async {
    pythonHost = newHost.trim().isEmpty ? "127.0.0.1" : newHost.trim();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString("python_host", pythonHost);
  }

  static Future<void> setPythonPort(int newPort) async {
    pythonPort = newPort <= 0 ? 8765 : newPort;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt("python_port", pythonPort);
  }

  static Future<void> setRecorderUrl(String newUrl) async {
    recorderUrl = newUrl.trim();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString("recorder_url", recorderUrl);
  }
}
