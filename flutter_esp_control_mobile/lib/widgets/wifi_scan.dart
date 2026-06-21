import 'package:flutter/material.dart';
import '../api/esp_http_api.dart';
import '../ui/ui_card.dart';
import 'wifi_bars.dart';

class WiFiScanCard extends StatefulWidget {
  final bool compact;

  const WiFiScanCard({super.key, this.compact = false});

  @override
  State<WiFiScanCard> createState() => _WiFiScanCardState();
}

class _WiFiScanCardState extends State<WiFiScanCard> {
  bool scanning = false;
  List<Map<String, dynamic>> aps = [];
  String error = "";

  Future<void> scan() async {
    setState(() {
      scanning = true;
      error = "";
      aps.clear();
    });

    try {
      aps = await EspHttpApi.wifiScan();
    } catch (e) {
      error = e.toString();
    }

    setState(() => scanning = false);
  }

  Future<void> connectDialog(String ssid) async {
    final controller = TextEditingController();
    bool connecting = false;

    await showDialog(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setStateDialog) {
          return AlertDialog(
            title: Text("連線到 $ssid"),
            content: TextField(
              controller: controller,
              obscureText: true,
              decoration: const InputDecoration(labelText: "Wi-Fi 密碼"),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text("取消"),
              ),
              ElevatedButton(
                onPressed: connecting
                    ? null
                    : () async {
                        setStateDialog(() => connecting = true);

                        try {
                          final ok = await EspHttpApi.wifiConnect(
                            ssid,
                            controller.text,
                          );

                          if (!context.mounted) return;

                          Navigator.pop(context);

                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(ok ? "已連線並儲存 $ssid" : "連線失敗"),
                            ),
                          );
                        } catch (e) {
                          setStateDialog(() => connecting = false);
                        }
                      },
                child: Text(connecting ? "連線中…" : "連線"),
              ),
            ],
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return UiCard(
      title: "選取 Wi-Fi",
      minHeight: widget.compact ? 120 : 180,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Spacer(),
              IconButton(
                tooltip: "掃描",
                onPressed: scanning ? null : scan,
                icon: scanning
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.refresh),
              ),
            ],
          ),
          const SizedBox(height: 8),
          if (error.isNotEmpty)
            Text("錯誤：$error", style: const TextStyle(color: Colors.red))
          else if (aps.isEmpty)
            const Text("尚未掃描")
          else
            ...aps.map((ap) {
              final apSsid = (ap['ssid'] ?? '').toString();
              final apRssi = (ap['rssi'] ?? 0) as int;

              return ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text(
                  apSsid,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                subtitle: Text("RSSI: $apRssi"),
                trailing: SizedBox(
                  width: 80, // ✅ 固定右側寬度，視覺會更整齊
                  child: Align(
                    alignment: Alignment.centerRight,
                    child: wifiBars(apRssi),
                  ),
                ),
                onTap: () => connectDialog(apSsid),
              );
            }),
        ],
      ),
    );
  }
}
