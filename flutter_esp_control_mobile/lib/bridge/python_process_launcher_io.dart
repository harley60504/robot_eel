import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';

class PythonLaunchResult {
  final bool ok;
  final String message;

  const PythonLaunchResult(this.ok, this.message);
}

class PythonProcessLauncher {
  static const String _backendFolderName = 'python_backend';

  static Process? _process;

  static Future<PythonLaunchResult> launch() async {
    if (!(Platform.isWindows || Platform.isMacOS || Platform.isLinux)) {
      return const PythonLaunchResult(
        false,
        'mobile apps cannot start a Python process on the PC',
      );
    }

    if (_process != null) {
      return const PythonLaunchResult(true, 'Python process already launched');
    }

    final workDir = _findBackendDirectory();
    if (workDir == null) {
      return const PythonLaunchResult(false, 'missing folder: python_backend');
    }

    final attempts = Platform.isWindows
        ? <_PythonCommand>[
            const _PythonCommand('python', [
              '-m',
              'uvicorn',
              'controller:app',
              '--host',
              '127.0.0.1',
              '--port',
              '8765',
            ]),
            const _PythonCommand('py', [
              '-3',
              '-m',
              'uvicorn',
              'controller:app',
              '--host',
              '127.0.0.1',
              '--port',
              '8765',
            ]),
          ]
        : <_PythonCommand>[
            const _PythonCommand('python3', [
              '-m',
              'uvicorn',
              'controller:app',
              '--host',
              '127.0.0.1',
              '--port',
              '8765',
            ]),
            const _PythonCommand('python', [
              '-m',
              'uvicorn',
              'controller:app',
              '--host',
              '127.0.0.1',
              '--port',
              '8765',
            ]),
          ];

    Object? lastError;
    for (final cmd in attempts) {
      try {
        final process = await Process.start(
          cmd.executable,
          cmd.arguments,
          workingDirectory: workDir.path,
          mode: ProcessStartMode.normal,
        );
        _process = process;
        unawaited(process.stdout
            .transform(systemEncoding.decoder)
            .listen((line) => debugPrint('[python] $line'))
            .asFuture<void>());
        unawaited(process.stderr
            .transform(systemEncoding.decoder)
            .listen((line) => debugPrint('[python] $line'))
            .asFuture<void>());
        unawaited(process.exitCode.then((_) => _process = null));
        return PythonLaunchResult(
          true,
          'launched ${cmd.executable} in ${workDir.path}',
        );
      } catch (e) {
        lastError = e;
      }
    }

    return PythonLaunchResult(false, 'launch failed: $lastError');
  }

  static Directory? _findBackendDirectory() {
    final currentDir = Directory.current;
    final exeDir = File(Platform.resolvedExecutable).parent;
    final candidates = <Directory>[
      Directory(
          '${currentDir.path}${Platform.pathSeparator}$_backendFolderName'),
      Directory('${exeDir.path}${Platform.pathSeparator}$_backendFolderName'),
      Directory(
        '${exeDir.parent.path}${Platform.pathSeparator}$_backendFolderName',
      ),
    ];

    for (final dir in candidates) {
      final controller = File(
        '${dir.path}${Platform.pathSeparator}controller.py',
      );
      if (dir.existsSync() && controller.existsSync()) {
        return dir;
      }
    }
    return null;
  }
}

class _PythonCommand {
  final String executable;
  final List<String> arguments;

  const _PythonCommand(this.executable, this.arguments);
}
