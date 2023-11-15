from urllib import request


def is_huggingface_available():
    try:
        request.urlopen('https://huggingface.co/', timeout=5)
        return True
    except request.URLError as err:
        return False


if __name__ == '__main__':
    print(is_huggingface_availble())
