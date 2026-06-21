import 'package:flutter/material.dart';
import '../api/python_api.dart';
import '../bridge/python_bridge.dart';
import '../bridge/python_process_launcher.dart';
import '../config.dart';
import '../ui/ui_card.dart';

class PythonPage extends StatefulWidget {
  final bool compact;
  final bool fillHeight;
  final bool embedded;

  const PythonPage({
    super.key,
    this.compact = false,
    this.fillHeight = false,
    this.embedded = false,
  });

  @override
  State<PythonPage> createState() => _PythonPageState();
}

class _PythonPageState extends State<PythonPage> {
  bool running = false;
  bool busy = false;
  bool loadingGaits = false;
  bool measuring = false;
  String selectedGait = "straight_rl";
  String outputMode = "cpg";
  List<Map<String, dynamic>> gaits = const [
    {"key": "straight_rl", "label": "Straight RL"},
    {"key": "left_turn_rl", "label": "Left Turn RL"},
    {"key": "left_spin_rl", "label": "Left Strong RL"},
    {"key": "right_turn_rl", "label": "Right Turn RL"},
    {"key": "right_spin_rl", "label": "Right Strong RL"},
  ];
  String logText = "";

  @override
  void initState() {
    super.initState();
    loadGaits();
  }

  void log(String s) {
    if (!mounted) return;
    setState(() => logText = "$s\n$logText");
  }

  Future<void> loadGaits() async {
    if (loadingGaits) return;
    setState(() => loadingGaits = true);
    final loaded = await PythonApi.gaits(pcHost: ApiConfig.pythonHost);
    if (!mounted) return;
    setState(() {
      loadingGaits = false;
      if (loaded.isEmpty) return;
      gaits = loaded;
      final selectedExists =
          loaded.any((item) => item["key"]?.toString() == selectedGait);
      if (!selectedExists) {
        selectedGait = loaded.first["key"]?.toString() ?? selectedGait;
      }
    });
  }

  Future<void> onStart() async {
    if (busy) return;
    final pcHost = ApiConfig.pythonHost;
    setState(() => busy = true);

    try {
      log("check python API ...");
      var ready = await PythonApi.ping(pcHost: pcHost);

      if (!ready) {
        log("python API offline, try local launch ...");
        final launch = await PythonProcessLauncher.launch();
        log(launch.message);

        if (launch.ok) {
          ready = await PythonApi.waitUntilReady(pcHost: pcHost);
          log("python API ready = $ready");
        }
      }

      if (!ready) {
        log("python API not ready, cannot start");
        return;
      }

      await loadGaits();
      if (!mounted) return;

      log("sync ESP32 host -> Python ...");
      final syncOk = await PythonBridge.syncEsp32HostToPython(pcHost: pcHost);
      if (!mounted) return;
      log("sync ok = $syncOk");

      log("set gait/output ...");
      final results = await Future.wait([
        PythonApi.setGait(pcHost: pcHost, gait: selectedGait),
        PythonApi.setOutputMode(pcHost: pcHost, outputMode: outputMode),
      ]);
      log("gait ok = ${results[0]}, output ok = ${results[1]}");

      log("start python...");
      final ok = await PythonApi.start(pcHost: pcHost);

      if (!mounted) return;
      setState(() => running = ok);
      log("start ok = $ok");
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  String gaitLabel(String key) {
    for (final item in gaits) {
      if (item["key"]?.toString() == key) {
        return item["label"]?.toString() ?? key;
      }
    }
    return key;
  }

  String gaitSubtitle(Map<String, dynamic> item) {
    final parts = <String>[];
    final ajoint = item["ajoint"];
    final frequency = item["frequency"];
    final lambda = item["lambda"];
    if (ajoint != null) parts.add("A=$ajoint");
    if (frequency != null) parts.add("f=$frequency");
    if (lambda != null) parts.add("lambda=$lambda");
    return parts.join("  ");
  }

  Future<void> showGaitSheet() async {
    if (running || busy) return;
    if (gaits.length <= 5) {
      await loadGaits();
      if (!mounted) return;
    }
    final searchCtrl = TextEditingController();
    final picked = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (context) {
        var query = "";
        return StatefulBuilder(
          builder: (context, setSheetState) {
            final q = query.trim().toLowerCase();
            final filtered = q.isEmpty
                ? gaits
                : gaits.where((item) {
                    final key = item["key"]?.toString().toLowerCase() ?? "";
                    final label = item["label"]?.toString().toLowerCase() ?? "";
                    return key.contains(q) || label.contains(q);
                  }).toList();
            return SafeArea(
              child: Padding(
                padding: EdgeInsets.only(
                  left: 16,
                  right: 16,
                  bottom: MediaQuery.of(context).viewInsets.bottom + 16,
                ),
                child: SizedBox(
                  height: MediaQuery.of(context).size.height * 0.78,
                  child: Column(
                    children: [
                      Row(
                        children: [
                          const Expanded(
                            child: Text(
                              "Select gait JSON",
                              style: TextStyle(
                                  fontSize: 18, fontWeight: FontWeight.w600),
                            ),
                          ),
                          IconButton(
                            tooltip: "Refresh",
                            onPressed: () async {
                              await loadGaits();
                              if (context.mounted) setSheetState(() {});
                            },
                            icon: const Icon(Icons.refresh),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      TextField(
                        controller: searchCtrl,
                        decoration: const InputDecoration(
                          labelText: "Search",
                          prefixIcon: Icon(Icons.search),
                        ),
                        onChanged: (value) =>
                            setSheetState(() => query = value),
                      ),
                      const SizedBox(height: 8),
                      Align(
                        alignment: Alignment.centerLeft,
                        child:
                            Text("${filtered.length} / ${gaits.length} gaits"),
                      ),
                      const SizedBox(height: 8),
                      Expanded(
                        child: ListView.separated(
                          itemCount: filtered.length,
                          separatorBuilder: (_, __) => const Divider(height: 1),
                          itemBuilder: (context, index) {
                            final item = filtered[index];
                            final key = item["key"]?.toString() ?? "";
                            final label = item["label"]?.toString() ?? key;
                            final subtitle = gaitSubtitle(item);
                            return ListTile(
                              selected: key == selectedGait,
                              title:
                                  Text(label, overflow: TextOverflow.ellipsis),
                              subtitle: Text(
                                subtitle.isEmpty ? key : "$key\n$subtitle",
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                              ),
                              trailing: key == selectedGait
                                  ? const Icon(Icons.check)
                                  : null,
                              onTap: () => Navigator.pop(context, key),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        );
      },
    );
    searchCtrl.dispose();
    if (!mounted || picked == null || picked.isEmpty) return;
    setState(() => selectedGait = picked);
  }

  Future<void> onStop() async {
    if (busy) return;
    final pcHost = ApiConfig.pythonHost;
    setState(() {
      busy = true;
      running = false;
    });

    try {
      final ok = await PythonApi.stop(pcHost: pcHost);
      log("stop ok = $ok");
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  Future<void> onMeasureToggle() async {
    final pcHost = ApiConfig.pythonHost;

    if (!measuring) {
      final ok = await PythonApi.measureOn(pcHost: pcHost);
      if (!mounted) return;

      setState(() => measuring = ok);
      log(ok ? "measure ON" : "measure ON failed");
    } else {
      final ok = await PythonApi.measureOff(pcHost: pcHost);
      if (!mounted) return;

      setState(() => measuring = ok ? false : true);
      log(ok ? "measure OFF" : "measure OFF failed");
    }
  }

  @override
  Widget build(BuildContext context) {
    final content = buildContent(context);

    if (widget.embedded) return content;

    return UiCard(
      title: "Python",
      minHeight: widget.fillHeight ? 0 : (widget.compact ? 260 : 340),
      fill: widget.fillHeight,
      child: content,
    );
  }

  Widget buildContent(BuildContext context) {
    final logBox = Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(8),
      ),
      child: SingleChildScrollView(
        child: Text(logText.isEmpty ? "(no logs)" : logText),
      ),
    );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: widget.fillHeight ? MainAxisSize.max : MainAxisSize.min,
      children: [
        Text(
          "Python API: ${ApiConfig.pythonHost}:${ApiConfig.pythonPort}",
          style: const TextStyle(color: Colors.white70),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: InkWell(
                onTap: running || busy ? null : showGaitSheet,
                child: InputDecorator(
                  decoration: InputDecoration(
                    labelText: "Gait JSON",
                    suffixIcon: loadingGaits
                        ? const Padding(
                            padding: EdgeInsets.all(14),
                            child: SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                          )
                        : const Icon(Icons.expand_more),
                  ),
                  child: Text(
                    gaitLabel(selectedGait),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: DropdownButtonFormField<String>(
                initialValue: outputMode,
                decoration: const InputDecoration(labelText: "Output"),
                items: const [
                  DropdownMenuItem(value: "angle", child: Text("Mode 3 Angle")),
                  DropdownMenuItem(value: "cpg", child: Text("Mode 1 CPG")),
                ],
                onChanged: running || busy
                    ? null
                    : (value) {
                        if (value == null) return;
                        setState(() => outputMode = value);
                      },
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: ElevatedButton(
                onPressed: running || busy ? null : onStart,
                child: FittedBox(child: Text(busy ? "Wait" : "Start")),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton(
                onPressed: running && !busy ? onStop : null,
                child: const FittedBox(child: Text("Stop")),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: OutlinedButton(
                onPressed: onMeasureToggle,
                child: FittedBox(
                  child: Text(measuring ? "Meas OFF" : "Meas ON"),
                ),
              ),
            ),
          ],
        ),
        SizedBox(height: widget.compact ? 10 : 16),
        const Text("Log"),
        const SizedBox(height: 6),
        if (widget.fillHeight)
          Expanded(child: logBox)
        else
          SizedBox(height: widget.compact ? 120 : 180, child: logBox),
      ],
    );
  }
}
