import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'dart:io' as io;
import 'package:web_socket_channel/web_socket_channel.dart';

class CameraStreamWS extends StatefulWidget {
  final String wsUrl;
  final bool initiallyPaused;
  final bool showFullscreenButton;

  const CameraStreamWS({
    super.key,
    required this.wsUrl,
    this.initiallyPaused = true,
    this.showFullscreenButton = true,
  });

  @override
  State<CameraStreamWS> createState() => _CameraStreamWSState();
}

class _CameraStreamWSState extends State<CameraStreamWS> {
  io.WebSocket? _socket;
  StreamSubscription? _socketSub;

  WebSocketChannel? _channel;
  StreamSubscription? _channelSub;

  Uint8List? frame;
  late bool paused;

  int frameCount = 0;
  double fps = 0;
  DateTime lastTime = DateTime.now();

  @override
  void initState() {
    super.initState();
    paused = widget.initiallyPaused;
    if (!paused) _connect();
  }

  Future<void> _connect() async {
    await _disconnect();
    if (kIsWeb) {
      _connectWeb();
    } else {
      await _connectMobile();
    }
  }

  Future<void> _disconnect() async {
    await _socketSub?.cancel();
    await _channelSub?.cancel();

    await _socket?.close();
    await _channel?.sink.close();

    _socketSub = null;
    _channelSub = null;
    _socket = null;
    _channel = null;
  }

  Future<void> togglePause() async {
    if (paused) {
      setState(() => paused = false);
      await _connect();
    } else {
      setState(() {
        paused = true;
        fps = 0;
        frameCount = 0;
      });
      await _disconnect();
    }
  }

  void openFullscreen() {
    Navigator.of(context).push(
      MaterialPageRoute(
        fullscreenDialog: true,
        builder: (_) {
          return Scaffold(
            backgroundColor: Colors.black,
            appBar: AppBar(
              backgroundColor: Colors.black,
              title: const Text("Camera"),
            ),
            body: SafeArea(
              child: CameraStreamWS(
                wsUrl: widget.wsUrl,
                initiallyPaused: false,
                showFullscreenButton: false,
              ),
            ),
          );
        },
      ),
    );
  }

  Future<void> _connectMobile() async {
    try {
      _socket = await io.WebSocket.connect(widget.wsUrl);

      _socketSub = _socket!.listen(
        (data) {
          if (!mounted) return;
          setState(() => frame = Uint8List.fromList(data as List<int>));
          _calcFPS();
        },
        onDone: () => debugPrint("Camera WS closed"),
        onError: (e) => debugPrint("Camera WS error: $e"),
      );
    } catch (e) {
      debugPrint("Camera WS connect failed: $e");
    }
  }

  void _connectWeb() {
    try {
      _channel = WebSocketChannel.connect(Uri.parse(widget.wsUrl));

      _channelSub = _channel!.stream.listen(
        (data) {
          if (!mounted) return;

          final bytes = (data is Uint8List)
              ? data
              : Uint8List.fromList(data as List<int>);

          setState(() => frame = bytes);
          _calcFPS();
        },
        onDone: () => debugPrint("Camera WS closed (web)"),
        onError: (e) => debugPrint("Camera WS error (web): $e"),
      );
    } catch (e) {
      debugPrint("Camera WS connect failed (web): $e");
    }
  }

  void _calcFPS() {
    frameCount++;
    final now = DateTime.now();
    final diff = now.difference(lastTime).inMilliseconds;

    if (diff >= 1000) {
      setState(() {
        fps = frameCount * 1000 / diff;
        frameCount = 0;
        lastTime = now;
      });
    }
  }

  @override
  void dispose() {
    _disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Stack(
        children: [
          Container(
            color: const Color(0xFF151A20),
            child: Center(
              child: paused
                  ? const Text(
                      "Camera paused",
                      style: TextStyle(color: Colors.white70),
                    )
                  : frame == null
                      ? const Text(
                          "Waiting for camera...",
                          style: TextStyle(color: Colors.white70),
                        )
                      : FittedBox(
                          fit: BoxFit.contain, // ✅ 不拉伸，只置中縮放
                          child: Image.memory(
                            frame!,
                            gaplessPlayback: true,
                          ),
                        ),
            ),
          ),
          Positioned(
            top: 8,
            left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(
                horizontal: 8,
                vertical: 6,
              ),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                "FPS: ${fps.toStringAsFixed(1)}",
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),
          Positioned(
            top: 8,
            right: 8,
            child: Row(
              children: [
                IconButton.filled(
                  tooltip: paused ? "播放" : "暫停",
                  onPressed: togglePause,
                  icon: Icon(paused ? Icons.play_arrow : Icons.pause),
                ),
                if (widget.showFullscreenButton) ...[
                  const SizedBox(width: 8),
                  IconButton.filled(
                    tooltip: "全螢幕",
                    onPressed: openFullscreen,
                    icon: const Icon(Icons.fullscreen),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
