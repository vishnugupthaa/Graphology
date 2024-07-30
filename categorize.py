def determine_baseline_angle(raw_baseline_angle):
    """
    Determines the baseline angle category and provides a comment.

    Args:
        raw_baseline_angle (float): The raw baseline angle value.

    Returns:
        tuple: (baseline_angle, comment)
    """
    if raw_baseline_angle >= 0.2:
        return 0, "DESCENDING"
    elif raw_baseline_angle <= -0.3:
        return 1, "ASCENDING"
    else:
        return 2, "STRAIGHT"


def determine_top_margin(raw_top_margin):
    """
    Determines the top margin category and provides a comment.

    Args:
        raw_top_margin (float): The raw top margin value.

    Returns:
        tuple: (top_margin, comment)
    """
    if raw_top_margin >= 1.7:
        return 0, "MEDIUM OR BIGGER"
    else:
        return 1, "NARROW"


def determine_letter_size(raw_letter_size):
    """
    Determines the letter size category and provides a comment.

    Args:
        raw_letter_size (float): The raw letter size value.

    Returns:
        tuple: (letter_size, comment)
    """
    if raw_letter_size >= 18.0:
        return 0, "BIG"
    elif raw_letter_size < 13.0:
        return 1, "SMALL"
    else:
        return 2, "MEDIUM"


def determine_line_spacing(raw_line_spacing):
    """
    Determines the line spacing category and provides a comment.

    Args:
        raw_line_spacing (float): The raw line spacing value.

    Returns:
        tuple: (line_spacing, comment)
    """
    if raw_line_spacing >= 3.5:
        return 0, "BIG"
    elif raw_line_spacing < 2.0:
        return 1, "SMALL"
    else:
        return 2, "MEDIUM"


def determine_word_spacing(raw_word_spacing):
    """
    Determines the word spacing category and provides a comment.

    Args:
        raw_word_spacing (float): The raw word spacing value.

    Returns:
        tuple: (word_spacing, comment)
    """
    if raw_word_spacing > 2.0:
        return 0, "BIG"
    elif raw_word_spacing < 1.2:
        return 1, "SMALL"
    else:
        return 2, "MEDIUM"


def determine_pen_pressure(raw_pen_pressure):
    """
    Determines the pen pressure category and provides a comment.

    Args:
        raw_pen_pressure (float): The raw pen pressure value.

    Returns:
        tuple: (pen_pressure, comment)
    """
    if raw_pen_pressure > 180.0:
        return 0, "HEAVY"
    elif raw_pen_pressure < 151.0:
        return 1, "LIGHT"
    else:
        return 2, "MEDIUM"


def determine_slant_angle(raw_slant_angle):
    """
    Determines the slant angle category and provides a comment.

    Args:
        raw_slant_angle (float): The raw slant angle value.

    Returns:
        tuple: (slant_angle, comment)
    """
    if raw_slant_angle in [-45.0, -30.0]:
        return 0, "EXTREMELY RECLINED"
    elif raw_slant_angle in [-15.0, -5.0]:
        return 1, "A LITTLE OR MODERATELY RECLINED"
    elif raw_slant_angle in [5.0, 15.0]:
        return 2, "A LITTLE INCLINED"
    elif raw_slant_angle == 30.0:
        return 3, "MODERATELY INCLINED"
    elif raw_slant_angle == 45.0:
        return 4, "EXTREMELY INCLINED"
    elif raw_slant_angle == 0.0:
        return 5, "STRAIGHT"
    else:
        return 6, "IRREGULAR"
