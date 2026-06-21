import 'package:flutter/material.dart';
import '../api/esp_http_api.dart';
import '../ui/ui_card.dart';

class WiFiCurrentCard extends StatefulWidget {
  const WiFiCurrentCard({super.key});

  @override
  State<WiFiCurrentCard> createState() => _WiFiCurrentCardState();
}

class _WiFiCurrentCardState extends State<WiFiCurrentCard> {
  bool loading = true;
  bool connected = false;
  String ssid = "-";
  String ip = "-";
  int rssi = 0;
  String error = "";

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
      final data = await EspHttpApi.wifiCurrent();

      setState(() {
        connected = data['connected'] ?? false;
        ssid = data['ssid'] ?? "-";
        ip = data['ip'] ?? "-";
        rssi = data['rssi'] ?? 0;
      });
    } catch (e) {
      error = e.toString();
    }

    setState(() => loading = false);
  }

  @override
  Widget build(BuildContext context) {
    return UiCard(
      title: "目前 Wi-Fi",
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (loading)
            const Text("讀取中…")
          else if (error.isNotEmpty)
            Text("錯誤：$error", style: const TextStyle(color: Colors.red))
          else if (!connected)
            const Text("未連線", style: TextStyle(color: Colors.red))
          else ...[
            Text("SSID：$ssid"),
            Text("IP：$ip"),
            Text("RSSI：$rssi dBm"),
          ],
          const SizedBox(height: 8),
          ElevatedButton(onPressed: refresh, child: const Text("重新讀取")),
        ],
      ),
    );
  }
}
