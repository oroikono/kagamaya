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
        self.assertIn('thumbnail opens these production notes', notes)
        self.assertNotIn('pretending to play a video', notes)
        self.assertNotIn('<iframe', readme)

    def test_real_recording_is_pending_not_claimed_complete(self):
        notes = (ROOT / 'docs/video.md').read_text()
        self.assertIn('not approved for upload', notes)
        self.assertIn('has not yet been received', notes)
        self.assertIn('(recording.md)', notes)
        self.assertIn('has not yet been rendered', notes)
        guide = (ROOT / 'docs/recording.md').read_text()
        self.assertIn('No recording has been received', guide)
        self.assertIn('separate approval', guide)
        self.assertIn('without background music', guide)


if __name__ == '__main__':
    unittest.main()
