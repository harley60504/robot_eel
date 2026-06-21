import '../api/python_api.dart';
import '../config.dart';

class PythonBridge {
  static Future<bool> syncEsp32HostToPython({
    required String pcHost,
  }) async {
    final espHost = ApiConfig.host;
    return await PythonApi.setEspHost(
      pcHost: pcHost,
      espHost: espHost,
    );
  }
}
