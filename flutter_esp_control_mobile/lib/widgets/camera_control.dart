import 'dart:async';
import 'package:flutter/material.dart';
import '../api/esp_api.dart';
import '../ui/ui_card.dart';

class CameraControlPanel extends StatefulWidget {
  final bool compact;
  final bool embedded;

  const CameraControlPanel({
    super.key,
    this.compact = false,
    this.embedded = false,
  });

  @override
  State<CameraControlPanel> createState() => _CameraControlPanelState();
}

class _CameraControlPanelState extends State<CameraControlPanel> {
  String resolution = "SVGA";
  double quality = 10;

  Timer? debounce;
  StreamSubscription? sub;

  final Map<String, int> frameSizeMap = {
    "UXGA": 11,
    "SXGA": 10,
    "SVGA": 7,
    "VGA": 6,
  };

  @override
  void initState() {
    super.initState();

    sub = WsControlApi.stream().listen((msg) {
      try {
        if (!mounted) return;
        if (msg is! Map) return;
        if (msg["type"] != "camera_param") return;

        setState(() {
          if (msg.containsKey("framesize")) {
            final rev = {for (final e in frameSizeMap.entries) e.value: e.key};
            resolution = rev[msg["framesize"]] ?? resolution;
          }
          if (msg.containsKey("quality")) {
            quality = (msg["quality"] as num).toDouble();
          }
        });
      } catch (e) {
        debugPrint("Camera WS parse error: $e");
      }
    });
  }

  @override
  void dispose() {
    debounce?.cancel();
    sub?.cancel();
    super.dispose();
  }

  void applyResolution(String value) {
    setState(() => resolution = value);
    WsControlApi.setCameraParam({"framesize": frameSizeMap[value]!});
  }

  void applyQuality(double v) {
    setState(() => quality = v);

    debounce?.cancel();
    debounce = Timer(const Duration(milliseconds: 300), () {
      WsControlApi.setCameraParam({"quality": v.toInt()});
    });
  }

  @override
  Widget build(BuildContext context) {
    final content = SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("解析度"),
          const SizedBox(height: 6),
          DropdownButton<String>(
            value: resolution,
            isExpanded: true,
            items: frameSizeMap.keys
                .map((k) => DropdownMenuItem(value: k, child: Text(k)))
                .toList(),
            onChanged: (v) {
              if (v == null) return;
              applyResolution(v);
            },
          ),
          const SizedBox(height: 16),
          Text("JPEG Quality: ${quality.toInt()}"),
          Slider(
            value: quality,
            min: 5,
            max: 60,
            divisions: 55,
            label: quality.toInt().toString(),
            onChanged: applyQuality,
          ),
        ],
      ),
    );

    if (widget.embedded) {
      return content;
    }

    return UiCard(
      title: "相機控制",
      minHeight: widget.compact ? 150 : 220,
      child: content,
    );
  }
}
