import 'package:flutter/material.dart';
import '../api/esp_api.dart';
import '../ui/ui_card.dart';
import '../ui/ui_layout.dart';

class ServoControlPanel extends StatefulWidget {
  final bool compact;
  final bool embedded;
  final bool centerCalibration;

  const ServoControlPanel({
    super.key,
    this.compact = false,
    this.embedded = false,
    this.centerCalibration = false,
  });

  @override
  State<ServoControlPanel> createState() => _ServoControlPanelState();
}

class _ServoControlPanelState extends State<ServoControlPanel> {
  static const int servoCount = 6;
  static const double minDeg = 0;
  static const double maxDeg = 240;

  final List<double> angles = List.filled(servoCount, 120.0);
  bool autoSend = false;
  bool _loadedCenterAngles = false;
  late final List<TextEditingController> _angleControllers;
  late final ScrollController _scrollController;

  @override
  void initState() {
    super.initState();
    _angleControllers = List.generate(
      servoCount,
      (i) => TextEditingController(text: angles[i].toStringAsFixed(1)),
    );
    _scrollController = ScrollController();
    _loadCenterAngles(WsControlApi.lastCtrlParams);
    WsControlApi.ctrlParamsNotifier.addListener(_onCtrlParamsChanged);
  }

  @override
  void dispose() {
    WsControlApi.ctrlParamsNotifier.removeListener(_onCtrlParamsChanged);
    for (final controller in _angleControllers) {
      controller.dispose();
    }
    _scrollController.dispose();
    super.dispose();
  }

  void _onCtrlParamsChanged() {
    _loadCenterAngles(WsControlApi.ctrlParamsNotifier.value);
  }

  void _loadCenterAngles(Map<String, dynamic>? params) {
    if (!widget.centerCalibration || params == null) return;
    if (_loadedCenterAngles) return;

    final centers = params['servoDefaultAngles'];
    if (centers is! List || centers.length < servoCount) return;

    _loadedCenterAngles = true;
    setState(() {
      for (int i = 0; i < servoCount; i++) {
        final value = centers[i];
        if (value is num) {
          angles[i] = value.toDouble().clamp(minDeg, maxDeg);
          _angleControllers[i].text = angles[i].toStringAsFixed(1);
        }
      }
    });
  }

  void _commitCenterInputs() {
    if (!widget.centerCalibration) return;

    for (int i = 0; i < servoCount; i++) {
      final parsed = double.tryParse(_angleControllers[i].text.trim());
      if (parsed == null) {
        _angleControllers[i].text = angles[i].toStringAsFixed(1);
        continue;
      }
      final clamped = parsed.clamp(minDeg, maxDeg);
      angles[i] = clamped;
      _angleControllers[i].text = clamped.toStringAsFixed(1);
    }
  }

  void sendAngles({bool saveCenter = false}) {
    if (widget.centerCalibration) {
      _commitCenterInputs();
      WsControlApi.setServoCenter(angles, save: saveCenter);
      return;
    }

    final mode = WsControlApi.lastCtrlParams?['mode'] ?? -1;
    if (mode != 3) return;
    WsControlApi.setAngle(angles);
  }

  void setAngleOnly(int index, double value) {
    final clamped = value.clamp(minDeg, maxDeg);
    setState(() {
      angles[index] = clamped;
      _angleControllers[index].text = clamped.toStringAsFixed(1);
    });
  }

  void setAngleFromInput(int index, String text) {
    final parsed = double.tryParse(text.trim());
    if (parsed == null) {
      _angleControllers[index].text = angles[index].toStringAsFixed(1);
      return;
    }

    setAngleOnly(index, parsed);
    if (autoSend) sendAngles();
  }

  @override
  Widget build(BuildContext context) {
    final content = buildContent();
    if (widget.embedded) return content;

    return UiCard(
      title: widget.centerCalibration
          ? "Offset \u4e2d\u5fc3\u89d2\u6821\u6b63"
          : "Angle \u63a7\u5236",
      minHeight: widget.compact ? 420 : 520,
      child: content,
    );
  }

  Widget buildContent() {
    return Scrollbar(
      controller: _scrollController,
      thumbVisibility: true,
      child: SingleChildScrollView(
        controller: _scrollController,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text("\u81ea\u52d5\u9001\u51fa"),
              subtitle: Text(widget.centerCalibration
                  ? "\u6ed1\u687f\u653e\u958b\u6642\u5957\u7528\u4e2d\u5fc3\u89d2\uff08\u4e0d\u5132\u5b58\uff09"
                  : "\u6ed1\u687f\u653e\u958b\u6642\u9001\u51fa set_angle"),
              value: autoSend,
              onChanged: (v) => setState(() => autoSend = v),
            ),
            const SizedBox(height: 8),
            ...List.generate(servoCount, (i) {
              return Padding(
                padding: EdgeInsets.only(bottom: widget.compact ? 8 : 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "${i + 1}  ${angles[i].toStringAsFixed(2)} deg",
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    Row(
                      children: [
                        Expanded(
                          child: Slider(
                            min: minDeg,
                            max: maxDeg,
                            value: angles[i].clamp(minDeg, maxDeg),
                            onChanged: (v) => setAngleOnly(i, v),
                            onChangeEnd: (_) {
                              if (autoSend) sendAngles();
                            },
                          ),
                        ),
                        SizedBox(
                          width: widget.centerCalibration
                              ? (widget.compact ? 78 : 90)
                              : (widget.compact ? 52 : 64),
                          child: widget.centerCalibration
                              ? TextField(
                                  controller: _angleControllers[i],
                                  keyboardType:
                                      const TextInputType.numberWithOptions(
                                    decimal: true,
                                    signed: false,
                                  ),
                                  textAlign: TextAlign.right,
                                  decoration: const InputDecoration(
                                    isDense: true,
                                    suffixText: "deg",
                                    contentPadding: EdgeInsets.symmetric(
                                      horizontal: 8,
                                      vertical: 8,
                                    ),
                                  ),
                                  onSubmitted: (text) =>
                                      setAngleFromInput(i, text),
                                  onEditingComplete: () => setAngleFromInput(
                                    i,
                                    _angleControllers[i].text,
                                  ),
                                )
                              : Text(
                                  angles[i].toStringAsFixed(1),
                                  textAlign: TextAlign.right,
                                ),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            }),
            const SizedBox(height: 4),
            Wrap(
              spacing: 12,
              runSpacing: 8,
              children: [
                SizedBox(
                  height: UiLayout.buttonHeight,
                  child: ElevatedButton(
                    onPressed: sendAngles,
                    child: Text(widget.centerCalibration
                        ? "\u5957\u7528\u4e2d\u5fc3\u89d2"
                        : "\u9001\u51fa angle"),
                  ),
                ),
                if (widget.centerCalibration)
                  SizedBox(
                    height: UiLayout.buttonHeight,
                    child: ElevatedButton.icon(
                      onPressed: () => sendAngles(saveCenter: true),
                      icon: const Icon(Icons.save),
                      label: const Text("\u5132\u5b58\u5230 ESP32"),
                    ),
                  ),
                SizedBox(
                  height: UiLayout.buttonHeight,
                  child: OutlinedButton(
                    onPressed: () {
                      setState(() {
                        for (int i = 0; i < servoCount; i++) {
                          angles[i] = 120.0;
                          _angleControllers[i].text = "120.0";
                        }
                      });
                      if (autoSend) sendAngles();
                    },
                    child: const Text("\u91cd\u8a2d 120 deg"),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
