from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]


class VideoPreviewTests(unittest.TestCase):
    def test_preview_assets_and_pending_upload_status(self):
        image = ROOT / 'assets/video/workshop-film-thumbnail.png'
        data = image.read_bytes()
        self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(struct.unpack('>II', data[16:24]), (1920, 1080))
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('assets/video/workshop-film-thumbnail.png', readme)
        self.assertIn('(docs/video.md)', readme)
        notes = (ROOT / 'docs/video.md').read_text()
        self.assertIn('listening review', notes)
        self.assertIn('No YouTube URL', notes)
        self.assertNotIn('<iframe', readme)


if __name__ == '__main__':
    unittest.main()
