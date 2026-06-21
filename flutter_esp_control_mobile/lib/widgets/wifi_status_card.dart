import 'package:flutter/material.dart';
import '../api/esp_http_api.dart';
import '../net/host_resolver.dart';
import '../net/wifi_info.dart';
import '../ui/ui_card.dart';

class WiFiStatusCard extends StatefulWidget {
  final bool compact;

  const WiFiStatusCard({super.key, this.compact = false});

  @override
  State<WiFiStatusCard> createState() => _WiFiStatusCardState();
}

class _WiFiStatusCardState extends State<WiFiStatusCard> {
  bool loading = true;
  String error = "";

  // ✅ 手機目前 SSID（顯示用）
  String phoneSsid = "-";
  bool phoneSsidOk = false;

  // current
  bool connected = false;
  String ssid = "-";
  String ip = "-";
  int rssi = 0;

  // saved list
  List<String> saved = [];

  @override
  void initState() {
    super.initState();

    // ✅ 先用 bootSsid 立刻顯示（不等 refresh）
    final boot = WifiInfo.bootSsid;
    phoneSsidOk = boot != null;
    phoneSsid = boot ?? "(開 App 時讀不到 SSID)";

    // ✅ 不要 initState 直接 refresh（太早）
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Future.delayed(const Duration(milliseconds: 250), () {
        if (mounted) refresh();
      });
    });
  }

  Future<void> refresh() async {
    setState(() {
      loading = true;
      error = "";
    });

    try {
      // ✅ 0) 手機 SSID：refresh 時才即時更新一次
      final s = await WifiInfo.getCurrentSsid();
      phoneSsidOk = s != null;
      phoneSsid = s ?? "(讀不到 SSID，請稍後再試)";

      // ✅ 1) current
      final current = await EspHttpApi.wifiCurrent();
      connected = current['connected'] ?? false;
      ssid = current['ssid'] ?? "-";
      ip = current['ip'] ?? "-";
      rssi = current['rssi'] ?? 0;

      // ✅ 2) saved list
      saved = await EspHttpApi.wifiSaved();

      // ✅ 3) 如果已連線且有拿到 STA IP → 更新快取
      if (connected && ip != "-") {
        await HostResolver.updateCachesByStaIp(ip);
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
      title: widget.compact ? "目前連線" : "Wi-Fi 狀態總覽",
      minHeight: widget.compact ? 92 : 180,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ✅ 手機 SSID 顯示（直接顯示 bootSsid → refresh 後更新）
          if (!widget.compact)
            Text(
              "目前 Wi-Fi：$phoneSsid",
              style: TextStyle(
                color: phoneSsidOk ? Colors.white : Colors.orange,
              ),
            ),

          if (!widget.compact) const SizedBox(height: 8),

          if (loading)
            const Text("讀取中…")
          else if (error.isNotEmpty)
            Text("錯誤：$error", style: const TextStyle(color: Colors.red))
          else if (widget.compact)
            ListTile(
              contentPadding: EdgeInsets.zero,
              leading: CircleAvatar(
                backgroundColor:
                    connected ? Colors.green.shade700 : Colors.grey,
                child: const Icon(Icons.wifi, color: Colors.white),
              ),
              title: Text(
                connected ? ssid : "未連線",
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              subtitle: Text(connected ? "$ip · $rssi dBm" : phoneSsid),
              trailing: IconButton(
                tooltip: "重新讀取",
                onPressed: refresh,
                icon: const Icon(Icons.refresh),
              ),
            )
          else ...[
            // ===== Current =====
            Text(
              "ESP32 連線狀態：${connected ? "已連線" : "未連線"}",
              style: TextStyle(
                color: connected ? Colors.white : Colors.red,
              ),
            ),
            SelectableText("ESP32 SSID：$ssid"),
            SelectableText("ESP32 IP：$ip"),
            Text("ESP32 RSSI：$rssi dBm"),

            const Divider(height: 24),

            // ===== Saved =====
            const Text("已儲存 Wi-Fi", style: TextStyle(fontSize: 16)),
            const SizedBox(height: 8),

            if (saved.isEmpty)
              const Text("尚未儲存")
            else
              ...saved.map(
                (s) => ListTile(
                  contentPadding: EdgeInsets.zero,
                  title: Text(s),
                ),
              ),
          ],

          if (!widget.compact) ...[
            const SizedBox(height: 8),
            ElevatedButton(
              onPressed: refresh,
              child: const Text("重新讀取"),
            ),
          ],
        ],
      ),
    );
  }
}
