import 'package:flutter/material.dart';

import 'api/esp_api.dart';
import 'api/python_api.dart';
import 'config.dart';
import 'net/host_resolver.dart';
import 'net/wifi_info.dart';
import 'pages/wifi_page.dart';
import 'ui/ui_layout.dart';
import 'widgets/camera_control.dart';
import 'widgets/control_dashboard.dart';
import 'widgets/python_rtsp_preview.dart';
import 'widgets/servo_table.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await ApiConfig.load();

  ErrorWidget.builder = (details) {
    return const Material(
      color: Color(0xFF050607),
      child: Center(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
            "\u756b\u9762\u8f09\u5165\u5931\u6557\uff0c\u8acb\u67e5\u770b debug console\u3002",
            style: TextStyle(color: Colors.white),
          ),
        ),
      ),
    );
  };

  runApp(const ESP32ControlApp());
}

class ESP32ControlApp extends StatelessWidget {
  const ESP32ControlApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF050607),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF27B5FF),
          brightness: Brightness.dark,
        ),
        cardTheme: CardThemeData(
          color: const Color(0xFF12161C),
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: const BorderSide(color: Color(0xFF252B33)),
          ),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF090B0E),
          foregroundColor: Colors.white,
          elevation: 0,
        ),
        inputDecorationTheme: const InputDecorationTheme(
          filled: true,
          fillColor: Color(0xFF0B0F14),
        ),
      ),
      home: const MainLayout(),
    );
  }
}

class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  int mode = -1;
  int activeMenu = 0;
  bool userSelectedMenu = false;

  @override
  void initState() {
    super.initState();

    final cached = WsControlApi.lastCtrlParams;
    if (cached != null) {
      mode = cached['mode'] ?? -1;
      activeMenu = mode >= 0 && mode <= 3 ? mode : 0;
    }

    WsControlApi.ctrlParamsNotifier.addListener(_onCtrlParams);
    _bootAndConnect();
  }

  Future<void> _bootAndConnect() async {
    try {
      await WifiInfo.initBootSsid();
      final r = await HostResolver.autoSelectHostEx();
      await ApiConfig.setHost(r.host, reason: r.reason);
      debugPrint("[BOOT] host=${ApiConfig.host}, via=${ApiConfig.hostReason}");
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint("[BOOT] fallback host=${ApiConfig.host}, error=$e");
    }

    Future.delayed(const Duration(seconds: 1), () {
      WsControlApi.ensureAllConnect();
    });
  }

  void _onCtrlParams() {
    final msg = WsControlApi.ctrlParamsNotifier.value;
    if (!mounted || msg == null) return;

    final newMode = msg['mode'] ?? -1;
    if (newMode != mode) {
      setState(() {
        mode = newMode;
        if (!userSelectedMenu && newMode >= 0 && newMode <= 3) {
          activeMenu = newMode;
        }
      });
    }
  }

  @override
  void dispose() {
    WsControlApi.ctrlParamsNotifier.removeListener(_onCtrlParams);
    super.dispose();
  }

  void openSettingsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.78,
          minChildSize: 0.36,
          maxChildSize: 0.92,
          builder: (context, scrollController) {
            return Container(
              decoration: const BoxDecoration(
                color: Color(0xFF0B0F14),
                borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
              ),
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
                  child: ListView(
                    controller: scrollController,
                    padding: const EdgeInsets.fromLTRB(14, 8, 14, 18),
                    children: [
                      Center(
                        child: Container(
                          width: 44,
                          height: 4,
                          margin: const EdgeInsets.only(bottom: 12),
                          decoration: BoxDecoration(
                            color: Colors.white24,
                            borderRadius: BorderRadius.circular(99),
                          ),
                        ),
                      ),
                      const _SettingsSheet(),
                    ],
                  ),
                ),
              ),
            );
          },
        );
      },
    );
  }

  Widget buildControlPanel({bool fillHeight = false}) {
    return ControlDashboard(
      selectedMode: activeMenu,
      fillHeight: fillHeight,
      onModeSelected: (m) {
        setState(() {
          activeMenu = m;
          userSelectedMenu = true;
        });
      },
    );
  }

  Widget buildPreviewArea({required bool compact}) {
    return const PythonRtspPreview();
  }

  @override
  Widget build(BuildContext context) {
    final isMobile =
        MediaQuery.of(context).size.width < UiLayout.mobileBreakpoint;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          "ESP32 \u63a7\u5236\u9762\u677f - ${ApiConfig.host} (${ApiConfig.hostReason})",
        ),
        actions: [
          IconButton(
            tooltip: "\u8a2d\u5b9a",
            onPressed: openSettingsSheet,
            icon: const Icon(Icons.settings),
          ),
        ],
      ),
      body: Padding(
        padding: UiLayout.pagePadding,
        child: isMobile
            ? SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    AspectRatio(
                      aspectRatio: 4 / 3,
                      child: buildPreviewArea(compact: true),
                    ),
                    const SizedBox(height: UiLayout.cardGap),
                    buildControlPanel(),
                  ],
                ),
              )
            : LayoutBuilder(
                builder: (context, constraints) {
                  final sideWidth = (constraints.maxWidth * 0.24)
                      .clamp(360.0, 470.0)
                      .toDouble();
                  final gap = constraints.maxWidth < 1100 ? 12.0 : UiLayout.gap;

                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Expanded(
                        child: buildPreviewArea(compact: false),
                      ),
                      SizedBox(width: gap),
                      SizedBox(
                        width: sideWidth,
                        child: buildControlPanel(fillHeight: true),
                      ),
                    ],
                  );
                },
              ),
      ),
    );
  }
}

class _SettingsSheet extends StatefulWidget {
  const _SettingsSheet();

  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  int selected = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      const WiFiPage(compact: true),
      const CameraControlPanel(compact: true, embedded: true),
      const ServoTable(compact: true),
      _ConnectionSettings(onSaved: () => setState(() {})),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                "\u8a2d\u5b9a",
                style: Theme.of(context).textTheme.titleLarge,
              ),
            ),
            IconButton(
              tooltip: "\u95dc\u9589",
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.close),
            ),
          ],
        ),
        const SizedBox(height: 8),
        SegmentedButton<int>(
          segments: const [
            ButtonSegment(
                value: 0, icon: Icon(Icons.wifi), label: Text("Wi-Fi")),
            ButtonSegment(
              value: 1,
              icon: Icon(Icons.tune),
              label: Text("\u76f8\u6a5f"),
            ),
            ButtonSegment(
              value: 2,
              icon: Icon(Icons.table_chart),
              label: Text("Servo"),
            ),
            ButtonSegment(
              value: 3,
              icon: Icon(Icons.link),
              label: Text("\u9023\u7dda"),
            ),
          ],
          selected: {selected},
          onSelectionChanged: (value) {
            setState(() => selected = value.first);
          },
          showSelectedIcon: false,
        ),
        const SizedBox(height: 12),
        pages[selected],
      ],
    );
  }
}

class _ConnectionSettings extends StatefulWidget {
  final VoidCallback onSaved;

  const _ConnectionSettings({required this.onSaved});

  @override
  State<_ConnectionSettings> createState() => _ConnectionSettingsState();
}

class _ConnectionSettingsState extends State<_ConnectionSettings> {
  late final TextEditingController pythonHostCtrl;
  late final TextEditingController pythonPortCtrl;
  late final TextEditingController recorderUrlCtrl;
  String status = "";

  @override
  void initState() {
    super.initState();
    pythonHostCtrl = TextEditingController(text: ApiConfig.pythonHost);
    pythonPortCtrl =
        TextEditingController(text: ApiConfig.pythonPort.toString());
    recorderUrlCtrl = TextEditingController(text: ApiConfig.recorderUrl);
  }

  @override
  void dispose() {
    pythonHostCtrl.dispose();
    pythonPortCtrl.dispose();
    recorderUrlCtrl.dispose();
    super.dispose();
  }

  Future<void> save() async {
    final port = int.tryParse(pythonPortCtrl.text.trim()) ?? 8765;
    await ApiConfig.setPythonHost(pythonHostCtrl.text);
    await ApiConfig.setPythonPort(port);
    await ApiConfig.setRecorderUrl(recorderUrlCtrl.text);

    final synced = await PythonApi.setRecorderUrl(
      pcHost: ApiConfig.pythonHost,
      recorderUrl: ApiConfig.recorderUrl,
    );

    widget.onSaved();
    if (!mounted) return;
    setState(() {
      status = synced
          ? "\u5df2\u5132\u5b58\u4e26\u540c\u6b65\u9304\u5f71 URL"
          : "\u5df2\u5132\u5b58\uff0cPython backend \u5c1a\u672a\u540c\u6b65";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        TextField(
          controller: pythonHostCtrl,
          decoration: const InputDecoration(
            labelText: "Python \u63a7\u5236 API Host",
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 10),
        TextField(
          controller: pythonPortCtrl,
          keyboardType: TextInputType.number,
          decoration: const InputDecoration(
            labelText: "Python \u63a7\u5236 API Port",
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 10),
        TextField(
          controller: recorderUrlCtrl,
          decoration: const InputDecoration(
            labelText: "Python \u9304\u5f71 RTSP URL",
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: UiLayout.buttonHeight,
          child: ElevatedButton.icon(
            onPressed: save,
            icon: const Icon(Icons.save),
            label: const Text("\u5132\u5b58\u9023\u7dda\u8a2d\u5b9a"),
          ),
        ),
        if (status.isNotEmpty) ...[
          const SizedBox(height: 10),
          Text(status, style: const TextStyle(color: Colors.white70)),
        ],
      ],
    );
  }
}
