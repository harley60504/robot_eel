import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';

const _releaseDirectory = r'C:\Users\ytyla\Documents\GitHub\robot_eel\Release';

Future<String?> saveCsvFile(String filename, Uint8List bytes) async {
  final pickerWritesBytes = Platform.isAndroid || Platform.isIOS;
  final initialDirectory =
      !pickerWritesBytes && Directory(_releaseDirectory).existsSync()
          ? _releaseDirectory
          : null;

  final path = await FilePicker.platform.saveFile(
    dialogTitle: "Save Servo CSV",
    fileName: filename,
    initialDirectory: initialDirectory,
    type: FileType.custom,
    allowedExtensions: ['csv'],
    bytes: pickerWritesBytes ? bytes : null,
  );

  if (path == null) return null;

  if (!pickerWritesBytes) {
    await File(path).writeAsBytes(bytes, flush: true);
  }

  return path;
}
