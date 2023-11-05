import re


class TransformationTokens:
    ATTACHMENT_SEPARATOR_TOKEN = "|"
    ATTACHMENT_POINT_TOKEN = "*"
    ATTACHMENT_POINT_NUM_REGEXP = r"\[{}:(\d+)\]".format(re.escape(ATTACHMENT_POINT_TOKEN))
    ATTACHMENT_POINT_REGEXP = r"(?:{0}|\[{0}[^\]]*\])".format(re.escape(ATTACHMENT_POINT_TOKEN))
    ATTACHMENT_POINT_NO_BRACKETS_REGEXP = r"(?<!\[){}".format(re.escape(ATTACHMENT_POINT_TOKEN))
