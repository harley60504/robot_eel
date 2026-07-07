import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';

Future<String?> saveCsvFile(String filename, Uint8List bytes) async {
  final path = await FilePicker.platform.saveFile(
    dialogTitle: "Save Servo CSV",
    fileName: filename,
    type: FileType.custom,
    allowedExtensions: ['csv'],
    bytes: bytes,
  );

  return path;
}
