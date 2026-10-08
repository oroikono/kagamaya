from pathlib import Path
import struct
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BRANDING = ROOT / 'assets/branding'
SVG = '{http://www.w3.org/2000/svg}'


class BrandingTests(unittest.TestCase):
    def test_vectors_are_accessible_and_self_contained(self):
        for filename in ('kagamaya-mark.svg', 'kagamaya-lockup.svg'):
            with self.subTest(filename=filename):
                tree = ET.parse(BRANDING / filename)
                root = tree.getroot()
                self.assertEqual(root.tag, SVG + 'svg')
                self.assertEqual(root.get('role'), 'img')
                self.assertIsNotNone(root.find(SVG + 'title'))
                self.assertIsNotNone(root.find(SVG + 'desc'))
                self.assertFalse(any(node.tag in {SVG + 'script', SVG + 'foreignObject', SVG + 'image'} for node in root.iter()))
                self.assertFalse(any(key.rsplit('}', 1)[-1].startswith('on') or key.rsplit('}', 1)[-1] == 'href'
                                     for node in root.iter() for key in node.attrib))

    def test_png_exports_have_expected_size(self):
        for filename, dimensions in (('kagamaya-mark.png', (512, 512)),
                                     ('kagamaya-lockup.png', (1560, 420)),
                                     ('kagamaya-perspective.png', (1254, 1254))):
            with self.subTest(filename=filename):
                data = (BRANDING / filename).read_bytes()
                self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
                self.assertEqual(data[12:16], b'IHDR')
                self.assertEqual(struct.unpack('>II', data[16:24]), dimensions)

    def test_readme_references_the_bundled_header(self):
        self.assertIn('assets/branding/kagamaya-perspective.png', (ROOT / 'README.md').read_text())

    def test_readme_quick_links_resolve(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('[Reproduce](#one-command-reproduction)', readme)
        self.assertIn('## One-command reproduction', readme)
        for relative in ('docs/artifact_links.md', 'docs/video.md', 'assets/branding/README.md'):
            with self.subTest(relative=relative):
                self.assertIn('](' + relative + ')', readme)
                self.assertTrue((ROOT / relative).is_file())

    def test_current_repository_links_use_project_name(self):
        for relative in ('STATUS.md', 'docs/artifact_links.md'):
            with self.subTest(relative=relative):
                text = (ROOT / relative).read_text()
                self.assertIn('https://github.com/oroikono/kagamaya', text)
                self.assertNotIn('https://github.com/oroikono/evidence-view-gaming-sim2science', text)
        self.assertIn('release/reproduction-and-hub-links', (ROOT / 'STATUS.md').read_text())


if __name__ == '__main__':
    unittest.main()
