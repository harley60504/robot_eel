import 'package:flutter/material.dart';
import '../api/esp_http_api.dart';
import '../net/host_resolver.dart';
import '../ui/ui_card.dart';

class WiFiSavedCard extends StatefulWidget {
  const WiFiSavedCard({super.key});

  @override
  State<WiFiSavedCard> createState() => _WiFiSavedCardState();
}

class _WiFiSavedCardState extends State<WiFiSavedCard> {
  bool loading = true;
  List<String> saved = [];
  String error = "";

  // 額外顯示：目前 ESP32 狀態
  String currentSsid = "-";
  String currentIp = "-";
  bool connected = false;

  @override
  void initState() {
    super.initState();
    refresh();
  }

  Future<void> refresh() async {
    setState(() {
      loading = true;
      error = "";
    });

    try {
      // ✅ 1) 讀 ESP32 已儲存 Wi-Fi
      saved = await EspHttpApi.wifiSaved();

      // ✅ 2) 讀目前連線狀態（拿到 STA IP）
      final info = await EspHttpApi.wifiCurrent();

      connected = info["connected"] == true;
      currentSsid = connected ? (info["ssid"]?.toString() ?? "-") : "-";
      currentIp = connected ? (info["ip"]?.toString() ?? "-") : "-";

      // ✅ 3) 如果已連線，存 SSID -> IP（存到手機端表格）
      if (connected && currentIp != "-") {
        await HostResolver.updateCachesByStaIp(currentIp);
      }
    } catch (e) {
      error = e.toString();
    }

    if (!mounted) return;
    setState(() => loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return UiCard(
      title: "已儲存 Wi-Fi",
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ✅ 顯示 ESP32 目前狀態 + IP（方便你確認存 IP 有沒有成功）
          Text("ESP32 連線狀態：${connected ? "已連線" : "未連線"}"),
          SelectableText("ESP32 SSID：$currentSsid"),
          SelectableText("ESP32 STA IP：$currentIp"),

          const SizedBox(height: 12),

          if (loading)
            const Text("讀取中…")
          else if (error.isNotEmpty)
            Text("錯誤：$error", style: const TextStyle(color: Colors.red))
          else if (saved.isEmpty)
            const Text("尚未儲存")
          else
            ...saved.map(
              (s) => ListTile(contentPadding: EdgeInsets.zero, title: Text(s)),
            ),

          const SizedBox(height: 8),
          ElevatedButton(onPressed: refresh, child: const Text("重新讀取")),
        ],
      ),
    );
  }
}
