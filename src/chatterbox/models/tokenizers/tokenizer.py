import importlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import threading
from unicodedata import category, normalize

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from russian_text_stresser.text_stresser import RussianTextStresser

# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)


class EnTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt

class ChineseCangjieConverter:
    """Convert Chinese characters to Cangjie tokens used by the model."""

    def __init__(self, model_dir: Path):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping(model_dir)
        self._init_segmenter()

    def _load_cangjie_mapping(self, model_dir: Path) -> None:
        cangjie_path = model_dir / CANGJIE_FILENAME
        if not cangjie_path.exists():
            cangjie_path = Path(
                hf_hub_download(
                    repo_id=CANGJIE_REPO_ID,
                    filename=CANGJIE_FILENAME,
                )
            )

        with cangjie_path.open("r", encoding="utf-8") as fp:
            for entry in json.load(fp):
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                self.cj2word.setdefault(code, []).append(word)

    def _init_segmenter(self) -> None:
        from spacy_pkuseg import pkuseg

        self.segmenter = pkuseg()

    def _cangjie_encode(self, glyph: str) -> str | None:
        code = self.word2cj.get(glyph)
        if code is None:
            return None
        index = self.cj2word[code].index(glyph)
        return code + (str(index) if index > 0 else "")

    def __call__(self, text: str) -> str:
        full_text = " ".join(self.segmenter.cut(text))
        output = []
        for glyph in full_text:
            if category(glyph) != "Lo":
                output.append(glyph)
                continue

            cangjie = self._cangjie_encode(glyph)
            if cangjie is None:
                output.append(glyph)
                continue

            output.extend(f"[cj_{code}]" for code in cangjie)
            output.append("[cj_.]")
        return "".join(output)



def add_russian_stress(text: str) -> str:
    """Add stress marks to Russian text."""
    try:
        stresser = RussianTextStresser()
        return stresser.stress_text(text)
    except Exception as exc:
        raise RuntimeError("Russian stress labeling failed") from exc


def initialize_russian_stresser() -> None:
    """Install, initialize, and verify Russian stress labeling."""
    health_check_text = normalize("NFKD", "твои слова ничего не значат.")
    stressed_text = add_russian_stress(health_check_text)
    if "\u0301" not in stressed_text:
        raise RuntimeError(
            "Russian stresser health check failed: no stress marks were produced"
        )
    logger.info("Russian stresser health check passed")


class MTLTokenizer:
    def __init__(self, vocab_file_path, text_preproc: str):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.text_preproc = text_preproc
        self.model_dir = Path(vocab_file_path).parent
        self.cangjie_converter = None
        self.cangjie_converter_lock = threading.Lock()
        self.check_vocabset_sot_eot()
        self.stresser = RussianTextStresser()

    def get_cangjie_converter(self) -> ChineseCangjieConverter:
        if self.cangjie_converter is None:
            with self.cangjie_converter_lock:
                if self.cangjie_converter is None:
                    self.cangjie_converter = ChineseCangjieConverter(self.model_dir)
        return self.cangjie_converter

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc
        
    def get_russian_stress(self, text: str) -> str:
        """Add stress marks to Russian text."""
        try:
            return self.stresser.stress_text(text)
        except Exception as exc:
            raise RuntimeError("Russian stress labeling failed") from exc
            
    def preprocess_text(
        self,
        raw_text: str,
        language_id: str = None,
    ):
        """Apply the text preprocessing mode used to train the model."""
        preprocessed_text = raw_text
        if "lower" in self.text_preproc:
            preprocessed_text = preprocessed_text.lower()

        if language_id == "zh":
            preprocessed_text = self.get_cangjie_converter()(preprocessed_text)
        elif language_id == "ru":
            preprocessed_text = self.get_russian_stress(preprocessed_text)

        if "NFKD" in self.text_preproc:
            preprocessed_text = normalize("NFKD", preprocessed_text)
        return preprocessed_text

    def text_to_tokens(
        self,
        text: str,
        language_id: str = None,
    ):
        text_tokens = self.encode(
            text,
            language_id=language_id,
        )
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(
        self,
        txt: str,
        language_id: str = None,
    ):
        """
        preprocess text > language-specific processing > prepend lang_id > encode

        The multilingual model expects language tokens like [en], [fr] to be prepended
        to condition the synthesis for the appropriate language.
        """
        txt = self.preprocess_text(
            txt,
            language_id=language_id,
        )

        # Prepend language token if provided
        if language_id:
            lang_token = f"[{language_id.lower()}]"
            txt = lang_token + txt

        txt = txt.replace(" ", SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(
            seq,
            skip_special_tokens=False
        )
        txt = txt.replace(" ", "")
        txt = txt.replace(SPACE, " ")
        txt = txt.replace(EOT, "")
        txt = txt.replace(UNK, "")
        return txt
