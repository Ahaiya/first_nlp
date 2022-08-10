import unicodedata


def show_parameters(logger, config, phase):
    title = phase + " config parameters"
    logger.info(title.center(40, '-'))
    for para in config:
        logger.info("---{} = {}".format(para, config[para]))
    return

# 数据清洗
def clean_data(orig_data):
    data_copy = orig_data.copy()
    for index in orig_data.index:
        line = orig_data[index].strip()      #去除首尾空格
        line = convert_to_unicode(line) #转换编码
        line = _clean_text(line)
        data_copy[index] = line
    return data_copy


def _clean_text(text):
    output = []
    for char in text:
        if _is_control(char) or _is_punctuation(char):      #删除 control 和 punctuation
            continue
        elif _is_whitespace(char):
            output.append(" ")            #要分词， 所以 不需要空格
            continue
        else:
            output.append(char)
    return "".join(output)


def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_punctuation(char):          #包括了 一些特殊标记
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 63) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 255)):
        return True
    if cp == 64:
        return False
    cat = unicodedata.category(char)
    if cat.startswith(("P", "S", "C")):
        return True
    return False


