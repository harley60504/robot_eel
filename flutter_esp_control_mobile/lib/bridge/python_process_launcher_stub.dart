class PythonLaunchResult {
  final bool ok;
  final String message;

  const PythonLaunchResult(this.ok, this.message);
}

class PythonProcessLauncher {
  static Future<PythonLaunchResult> launch() async {
    return const PythonLaunchResult(
      false,
      'local Python launch is not supported on this platform',
    );
  }
}
