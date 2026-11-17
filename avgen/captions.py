"""Subtitles: groups the narration's word timings into short captions and
writes them as an .ass subtitle file, which ffmpeg burns into the video, or
as a plain .srt file (saved next to each narration, for YouTube captions).

Everything about how they look comes from the "subtitles" part of the
"assemble" config, e.g. blue text with a yellow border:

    "subtitles": {
      "enabled": true,
      "min_words": 2, "max_words": 4,
      "style": {"color": "#0000FF", "outline_color": "#FFFF00"}
    }

Style settings (any left out use DEFAULT_STYLE below):
  font             font name; must be installed where ffmpeg runs
  font_size        in pixels of the final video (1920 high by default)
  bold, italic     true / false
  uppercase        true shows captions in CAPITALS
  color            text colour, "#RRGGBB", or "#RRGGBBAA" with AA the opacity (FF = solid)
  outline_color    border colour around each letter
  outline_width    border thickness in pixels (0 = none)
  shadow_color     drop shadow colour (a see-through "#RRGGBBAA" works well)
  shadow_depth     shadow distance in pixels (0 = none)
  box              true puts the text on a solid box of `box_color` instead of an outline
  box_color        the box's colour
  position         "bottom", "middle" or "top"
  margin_vertical  distance in pixels from the bottom (or top) edge
  margin_horizontal  space in pixels kept free on the left and right
  letter_spacing   extra pixels between letters
"""

# How captions look when the config doesn't say: big, bold, white with a
# black outline, in the lower third of a 1080x1920 video.
DEFAULT_STYLE = {
    "font": "DejaVu Sans",
    "font_size": 84,
    "bold": True,
    "italic": False,
    "uppercase": False,
    "color": "#FFFFFF",
    "outline_color": "#000000",
    "outline_width": 6,
    "shadow_color": "#00000080",
    "shadow_depth": 3,
    "box": False,
    "box_color": "#000000B0",
    "position": "bottom",
    "margin_vertical": 520,
    "margin_horizontal": 80,
    "letter_spacing": 0,
}

# How words are grouped when the config doesn't say: 2 to 4 per caption,
# and a new caption after a pause longer than this many seconds.
DEFAULT_MIN_WORDS = 2
DEFAULT_MAX_WORDS = 4
DEFAULT_MAX_PAUSE_SECONDS = 0.6

# A caption stays up until the next one starts if the gap is shorter than
# this, so the text doesn't flicker off between words.
JOIN_GAP_SECONDS = 0.35

POSITIONS = {"bottom": 2, "middle": 5, "top": 8}  # .ass "Alignment" values (centred horizontally)
SENTENCE_END = (".", "!", "?", ",", ";", ":")


def group_words(words, min_words=DEFAULT_MIN_WORDS, max_words=DEFAULT_MAX_WORDS,
                max_pause_seconds=DEFAULT_MAX_PAUSE_SECONDS):
    """Group word timings into captions of `min_words` to `max_words` words.

    A caption ends when it has `max_words` words, or (once it has at least
    `min_words`) at the end of a sentence or before a pause. Returns
    [{"text": "the town that", "start": 0.0, "end": 0.9}, ...].
    """
    if min_words < 1 or max_words < min_words:
        raise ValueError(f"need 1 <= min_words <= max_words, got {min_words} and {max_words}")
    words = [word for word in words if str(word.get("word", "")).strip()]
    captions, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        following = words[i + 1] if i + 1 < len(words) else None
        sentence_ends = str(word["word"]).strip().endswith(SENTENCE_END)
        pause = following is not None and following["start"] - word["end"] > max_pause_seconds
        if (following is None or len(current) >= max_words
                or (len(current) >= min_words and (sentence_ends or pause))):
            captions.append({"text": " ".join(str(w["word"]).strip() for w in current),
                             "start": current[0]["start"], "end": current[-1]["end"]})
            current = []

    for caption, following in zip(captions, captions[1:]):
        if following["start"] - caption["end"] < JOIN_GAP_SECONDS:
            caption["end"] = following["start"]
    return captions


def to_ass(captions, style=None, width=1080, height=1920):
    """The captions as an .ass subtitle file (text), styled by `style` over DEFAULT_STYLE."""
    s = {**DEFAULT_STYLE, **(style or {})}
    if s["position"] not in POSITIONS:
        raise ValueError(f"subtitle position must be one of {sorted(POSITIONS)}, got {s['position']!r}")
    border_style = 3 if s["box"] else 1  # 3 = text on a box, 1 = outline and shadow
    outline_colour = s["box_color"] if s["box"] else s["outline_color"]
    style_line = ",".join(str(value) for value in [
        "Caption", s["font"], s["font_size"],
        ass_colour(s["color"]), ass_colour(s["color"]), ass_colour(outline_colour), ass_colour(s["shadow_color"]),
        -1 if s["bold"] else 0, -1 if s["italic"] else 0, 0, 0,  # bold, italic, underline, strike-out
        100, 100, s["letter_spacing"], 0,                        # scale x, scale y, spacing, angle
        border_style, s["outline_width"], s["shadow_depth"], POSITIONS[s["position"]],
        s["margin_horizontal"], s["margin_horizontal"], s["margin_vertical"], 1,
    ])
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: {style_line}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for caption in captions:
        text = caption["text"].upper() if s["uppercase"] else caption["text"]
        lines.append(f"Dialogue: 0,{_ass_time(caption['start'])},{_ass_time(caption['end'])},Caption,,0,0,0,,"
                     f"{_ass_text(text)}")
    return "\n".join(lines) + "\n"


def to_srt(captions):
    """The captions as an .srt file (text), the plain format YouTube and most players accept."""
    blocks = [f"{number}\n{_srt_time(caption['start'])} --> {_srt_time(caption['end'])}\n{caption['text']}\n"
              for number, caption in enumerate(captions, start=1)]
    return "\n".join(blocks)


def ass_colour(value):
    """"#RRGGBB" or "#RRGGBBAA" (AA = opacity) -> .ass "&HAABBGGRR" (AA = transparency)."""
    hex_digits = str(value).lstrip("#")
    if len(hex_digits) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in hex_digits):
        raise ValueError(f"colours look like '#RRGGBB' or '#RRGGBBAA', got {value!r}")
    red, green, blue = hex_digits[0:2], hex_digits[2:4], hex_digits[4:6]
    opacity = int(hex_digits[6:8], 16) if len(hex_digits) == 8 else 255
    return f"&H{255 - opacity:02X}{blue}{green}{red}".upper()


def _ass_time(seconds):
    """e.g. 75.456 -> "0:01:15.46"."""
    centis = int(round(max(seconds, 0) * 100))
    hours, centis = divmod(centis, 360000)
    minutes, centis = divmod(centis, 6000)
    return f"{hours}:{minutes:02d}:{centis // 100:02d}.{centis % 100:02d}"


def _srt_time(seconds):
    """e.g. 75.456 -> "00:01:15,456"."""
    millis = int(round(max(seconds, 0) * 1000))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    return f"{hours:02d}:{minutes:02d}:{millis // 1000:02d},{millis % 1000:03d}"


def _ass_text(text):
    """Text safe inside an .ass line: no style codes, no line breaks."""
    return " ".join(text.replace("\\", "/").replace("{", "(").replace("}", ")").split())
