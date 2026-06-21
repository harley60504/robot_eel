import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../config.dart';

const enableWsDebug = false;

class WsControlApi {
  static WebSocketChannel? _ws;

  static final StreamController<dynamic> _controller =
      StreamController<dynamic>.broadcast();

  static Timer? _retryTimer;
  static int _retryMs = 500;
  static String? _connectedUrl;

  // ===== latency =====
  static int _seq = 0;

  static int _nowMs() => DateTime.now().millisecondsSinceEpoch;

  // ==============================
  // Cache / notifier
  // ==============================
  static Map<String, dynamic>? lastCtrlParams;

  static final ValueNotifier<Map<String, dynamic>?> ctrlParamsNotifier =
      ValueNotifier<Map<String, dynamic>?>(null);

  // ==============================
  // Stream
  // ==============================
  static Stream<dynamic> stream() {
    ensureConnect();
    return _controller.stream;
  }

  // ==============================
  // Connection
  // ==============================
  static void ensureConnect() {
    final url = ApiConfig.wsControlUrl;
    if (_ws != null && _connectedUrl == url) return;

    disconnect();
    _connect(url);
  }

  static void disconnect() {
    if (enableWsDebug) print("[WS] disconnect");

    _retryTimer?.cancel();
    _retryTimer = null;

    try {
      _ws?.sink.close();
    } catch (_) {}

    _ws = null;
    _connectedUrl = null;
  }

  static void _connect(String url) {
    if (enableWsDebug) print("[WS] connecting → $url");

    try {
      _ws = WebSocketChannel.connect(Uri.parse(url));
      _connectedUrl = url;
      _retryMs = 500;

      _ws!.stream.listen(
        (msg) {
          try {
            if (enableWsDebug) print("[WS RX] $msg");
            final decoded = jsonDecode(msg);

            _controller.add(decoded);

            if (decoded is Map && decoded["type"] == "ctrl_params") {
              lastCtrlParams = Map<String, dynamic>.from(decoded);
              ctrlParamsNotifier.value = lastCtrlParams;
            }
          } catch (e) {
            if (enableWsDebug) print("[WS] json decode error: $e");
          }
        },
        onDone: _handleDisconnectAndRetry,
        onError: (_) => _handleDisconnectAndRetry(),
        cancelOnError: true,
      );
    } catch (_) {
      _handleDisconnectAndRetry();
    }
  }

  static void _handleDisconnectAndRetry() {
    disconnect();
    _retryTimer?.cancel();

    final delay = Duration(milliseconds: _retryMs);
    if (enableWsDebug) {
      print("[WS] retry in ${delay.inMilliseconds}ms");
    }

    _retryTimer = Timer(delay, () {
      _retryTimer = null;
      _retryMs = (_retryMs * 2).clamp(500, 10000);
      ensureConnect();
    });
  }

  // ==============================
  // Send
  // ==============================
  static void send(Map<String, dynamic> body) {
    ensureConnect();
    if (_ws == null) return;

    final text = jsonEncode(body);
    if (enableWsDebug) print("[WS TX] $text");

    try {
      _ws!.sink.add(text);
    } catch (_) {
      _handleDisconnectAndRetry();
    }
  }

  // ==============================
  // API（全部支援 seq / ts_ms）
  // ==============================

  static void setParam(Map<String, dynamic> p) {
    final seq = _seq++;
    final ts = _nowMs();

    send({
      "cmd": "set_param",
      "seq": seq,
      "ts_ms": ts,
      ...p,
    });
  }

  static void setAngle(List<double> angles) {
    final seq = _seq++;
    final ts = _nowMs();

    send({
      "cmd": "set_angle",
      "seq": seq,
      "ts_ms": ts,
      "angles": angles,
    });
  }

  static void setServoCenter(List<double> angles, {bool save = false}) {
    final seq = _seq++;
    final ts = _nowMs();

    send({
      "cmd": "set_servo_center",
      "seq": seq,
      "ts_ms": ts,
      "save": save,
      "angles": angles,
    });
  }

  static void setCameraParam(Map<String, dynamic> p) =>
      send({"cmd": "camera_param", ...p});

  // ==============================
  // WiFi
  // ==============================
  static void wifiStatus() => send({"cmd": "wifi_status"});
  static void wifiScan() => send({"cmd": "wifi_scan"});
  static void wifiList() => send({"cmd": "wifi_list"});

  static void wifiConnect(String ssid, String pass) =>
      send({"cmd": "wifi_connect", "ssid": ssid, "pass": pass});

  static void wifiSave(String ssid, String pass) =>
      send({"cmd": "wifi_save", "ssid": ssid, "pass": pass});

  static void wifiDelete(String ssid) =>
      send({"cmd": "wifi_delete", "ssid": ssid});
}
