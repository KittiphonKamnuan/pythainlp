# SPDX-FileCopyrightText: 2016-2026 PyThaiNLP Project
# SPDX-FileType: SOURCE
# SPDX-License-Identifier: Apache-2.0
"""
Characterization tests for pythainlp.transliterate.royin.

Expected values were recorded from the implementation before refactoring
of _replace_consonants.
"""

import unittest

from pythainlp.transliterate.royin import _replace_consonants, _romanize

# (Thai word, expected romanization) pairs
ROMANIZE_CASES: tuple[tuple[str, str], ...] = (
    ("นฟูรรมฃื", "nafuamkhue"),
    ("หมลฮ๊ร", "mlahon"),
    ("อ่ช้รร", "chan"),
    ("ผว", "pho"),
    ("็", "็"),
    ("ห", ""),
    ("อชปโอฟว", "chapaop"),
    ("เ", "เ"),
    ("รยปฮดก", "rayapahadok"),
    ("ฟี", "fi"),
    ("ธอเง", "thonge"),
    ("ซซ๋แีฐ", "sasแีt"),
    ("เ๊แแ", "เแแ"),
    ("ฃว้", "kho"),
    ("มฑ", "mot"),
    ("รร", "ron"),
    ("ฤวซจซไษ", "ruewasachasasai"),
    ("มเไซอ", "masoeai"),
    ("ฟ้ิ", "fi"),
    ("๋รฤฉู", "rarueachu"),
    ("ลจทชฐพอ", "lachathachathapho"),
    ("๊ฐ", "th"),
    ("พู", "phu"),
    ("ล", "l"),
    ("งะรฐใ", "nganthใ"),
    ("ฅ", "kh"),
    ("ดณฉรท", "danachrot"),
    ("ปธ", "pot"),
    ("ูปรร", "ูpan"),
    ("ไ์กฝใึ", "kafaiใึ"),
    ("ฉแำอโฟ", "chแำfo"),
    ("ฉำร๋", "chamn"),
    ("ิ์ฬค่ข", "ิnkk"),
    ("ฑษูตธี", "thasutthi"),
    ("ซ", "s"),
    ("ุริชฏรรผ", "ุrittap"),
    ("", ""),
    ("กรรม", "kam"),
    ("กรร", "kan"),
    ("หมา", "ma"),
    ("หนู", "nu"),
    ("ฤ", "rue"),
    ("ฤษี", "rueasi"),
    ("คฤห", "khrue"),
    ("หรร", "an"),
    ("กร", "kon"),
    ("ขรร", "khan"),
    ("ศรี", "sri"),
    ("สวัสดี", "swatdi"),
    ("เหลว", "en"),
)

# (word, consonants, expected) for direct calls
DIRECT_CASES: tuple[tuple[str, str, str], ...] = (
    ("กรม", "กรม", "krom"),
    ("ก", "กข", "k"),
    ("ก", "กขค", "k"),
    # The output follows the word, not the consonants argument.
    ("อ", "ก", ""),
    ("ห", "ห", "h"),
    ("หก", "หก", "k"),
    ("กรร", "กน", "kan"),
    ("ากร", "กร", "าkn"),
)


class RoyinReplaceConsonantsTestCase(unittest.TestCase):
    def test_romanize_golden(self):
        for word, expected in ROMANIZE_CASES:
            with self.subTest(word=word):
                self.assertEqual(_romanize(word), expected)

    def test_replace_consonants_direct(self):
        for word, consonants, expected in DIRECT_CASES:
            with self.subTest(word=word, consonants=consonants):
                self.assertEqual(
                    _replace_consonants(word, consonants), expected
                )

    def test_empty_consonants_returns_word(self):
        self.assertEqual(_replace_consonants("abc", ""), "abc")
        self.assertEqual(_replace_consonants("", ""), "")

    def test_empty_word(self):
        self.assertEqual(_replace_consonants("", "ก"), "")

    def test_consonants_shorter_than_word(self):
        # The consonants argument no longer drives the index, so a short
        # one does not raise.
        for word, consonants, expected in (
            ("กข", "ก", "kok"),
            ("กรรม", "กม", "kam"),
        ):
            with self.subTest(word=word):
                self.assertEqual(
                    _replace_consonants(word, consonants), expected
                )

    def test_extra_consonants_ignored(self):
        self.assertEqual(_replace_consonants("ก", "กขค"), "k")


if __name__ == "__main__":
    unittest.main()
