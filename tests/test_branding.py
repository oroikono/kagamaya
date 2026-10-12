from pathlib import Path
from html.parser import HTMLParser
import struct
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BRANDING = ROOT / 'assets/branding'
SVG = '{http://www.w3.org/2000/svg}'


class BrandingTests(unittest.TestCase):
    def test_platform_icons_are_accessible_and_self_contained(self):
        for filename in ('huggingface.svg', 'youtube.svg'):
            with self.subTest(filename=filename):
                root = ET.parse(ROOT / 'assets/platforms' / filename).getroot()
                self.assertEqual(root.tag, SVG + 'svg')
                self.assertEqual(root.get('role'), 'img')
                self.assertTrue(root.find(SVG + 'title').text)
                self.assertEqual(root.get('viewBox'), '0 0 24 24')
                self.assertFalse(any(node.tag in {SVG + 'script', SVG + 'foreignObject', SVG + 'image'}
                                     for node in root.iter()))
                self.assertFalse(any(key.rsplit('}', 1)[-1].startswith('on') or key.rsplit('}', 1)[-1] == 'href'
                                     for node in root.iter() for key in node.attrib))

    def test_platform_icons_link_to_actual_destinations(self):
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.current = None
                self.icons = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'a':
                    self.current = attrs.get('href')
                if tag == 'img' and attrs.get('src', '').startswith('assets/platforms/'):
                    self.icons.append((self.current, attrs['src'], attrs.get('alt')))

            def handle_endtag(self, tag):
                if tag == 'a':
                    self.current = None

        opening = (ROOT / 'README.md').read_text().split('Accepted at', 1)[0]
        parser = Links()
        parser.feed(opening)
        self.assertEqual(parser.icons, [
            ('https://huggingface.co/datasets/oroikono/EvidenceViewBench-Dev', 'assets/platforms/huggingface.svg', 'Hugging Face'),
            ('https://huggingface.co/datasets/oroikono/VerifierExtension-Results', 'assets/platforms/huggingface.svg', 'Hugging Face'),
            ('docs/video.md', 'assets/platforms/youtube.svg', 'YouTube'),
        ])
        self.assertIn('YouTube — coming soon', opening)
        self.assertNotIn('youtube.com/watch', opening)

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

    def test_name_spelling_and_meaning_are_in_opening(self):
        opening = (ROOT / 'README.md').read_text().split('[Reproduce]', 1)[0]
        for phrase in ('K-A-G-A-M-A-Y-A', 'kah-gah-MAH-yah', 'kagami', 'māyā',
                       'mirror of appearances', 'not a literal translation'):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, opening)

    def test_hugging_face_links_are_prominent_and_scoped(self):
        readme = (ROOT / 'README.md').read_text()
        opening = readme.split('Accepted at', 1)[0]
        for repo in ('EvidenceViewBench-Dev', 'VerifierExtension-Results'):
            self.assertIn('https://huggingface.co/datasets/oroikono/' + repo, opening)
        self.assertIn('Both are private', opening)
        self.assertIn('completed analyses', readme)
        self.assertIn('preserved failed', readme)
        self.assertNotIn('not completed new results', readme)
        notes = (ROOT / 'docs/artifact_links.md').read_text()
        for phrase in ('completed analyses', 'preserved failed runs',
                       '5 October integration/repair', '7 October construction',
                       'scientific audit blocked'):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, notes)

    def test_readme_quick_links_resolve(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('[Reproduce](#quick-start)', readme)
        self.assertIn('## Quick start', readme)
        for relative in ('docs/artifact_links.md', 'docs/video.md', 'assets/branding/README.md'):
            with self.subTest(relative=relative):
                self.assertIn('](' + relative + ')', readme)
                self.assertTrue((ROOT / relative).is_file())

    def test_workshop_status_and_pending_paper_url_are_clear(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('https://www.sim2science.com/', readme)
        self.assertIn('a **NeurIPS 2026 workshop**.', readme)
        self.assertIn('direct OpenReview paper link is pending verification', readme)
        opening = readme.split('## Quick start', 1)[0]
        self.assertIn('historical v19 paper source', opening)
        self.assertIn('not the current camera-ready release', opening)
        notes = (ROOT / 'docs/artifact_links.md').read_text()
        self.assertIn('Workshop/Sim2Sci', notes)
        self.assertIn('neither venue link is presented as a paper PDF link', notes)

    def test_current_repository_links_use_project_name(self):
        for relative in ('STATUS.md', 'docs/artifact_links.md'):
            with self.subTest(relative=relative):
                text = (ROOT / relative).read_text()
                self.assertIn('https://github.com/oroikono/kagamaya', text)
                self.assertNotIn('https://github.com/oroikono/evidence-view-gaming-sim2science', text)
        self.assertIn('release/reproduction-and-hub-links', (ROOT / 'STATUS.md').read_text())
        self.assertIn('On 8 October 2026', (ROOT / 'STATUS.md').read_text())
        self.assertNotIn('active cleanup branch', (ROOT / 'STATUS.md').read_text())

    def test_quick_start_checks_manifest_before_recount_and_film(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertLess(readme.index('python3 scripts/verify_manifest.py'),
                        readme.index('python3 scripts/reproduce_findings.py'))
        self.assertLess(readme.index('## Quick start'), readme.index('## Workshop film'))
        for phrase in ("Python's standard library", '33/33',
                       'output/reproduction/code_crosscheck.json',
                       'does not rerun solvers', 'network or model-provider calls'):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, readme)

    def test_results_are_labeled_and_privacy_claim_is_scoped(self):
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('1,435 correct, 3 false, and 362 inconclusive', readme)
        self.assertIn('not independent replications', readme)
        self.assertNotIn('No credentials are stored here', readme)
        self.assertIn('docs/privacy_review_20261012.md', readme)
        self.assertIn('not an exhaustive guarantee', readme)


if __name__ == '__main__':
    unittest.main()
